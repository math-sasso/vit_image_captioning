import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
import sacrebleu

class CaptionModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Initializes Decoder + LM Head
        self.t5 = T5ForConditionalGeneration.from_pretrained(self.hparams.t5_prefix)
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True,
                                     representation_size=768)

        # Maches dimensions between EfficientNet and T5
        # self.effnet_to_t5_bridge = nn.Conv2d(
        #     in_channels=1280, out_channels=self.t5.model_dim, kernel_size=1)
                
        # Não está passando tokenizer pro Cuda quando usa o hparams
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_prefix)

        self.maybe_freeze_params()


    def maybe_freeze_params(self):
        """Optionally freezes params.

        """
        # Encoder T5
        if self.hparams.freeze_t5_encoder:
            print("Freezing params for t5 encoder:")
            for name, param  in self.t5.encoder.named_parameters():
                if name not in ['shared.weight', 'embed_tokens.weight']:
                    # logging.debug(f'\t{name}')
                    param.requires_grad = False
        # T5 decoder
        if self.hparams.freeze_t5_decoder:
            print("Freezing params for t5 decoder:")
            for name, param  in self.t5.decoder.named_parameters():
                if name not in ['shared.weight', 'embed_tokens.weight']:
                    # logging.debug(f'\t{name}')
                    param.requires_grad = False

        # Embeddings T5
        if self.hparams.freeze_t5_embeddings:
            print(f'Freezing t5 embeddings weights')
            self.t5.shared.weight.requires_grad = False

        # EfficientNet
        if self.hparams.freeze_image_encoder:
            print("Freezing Image encoder params:")
            for name, param  in self.vit.named_parameters():
                # logging.debug(f'\t{name}')
                param.requires_grad = False


    def encode_image(self, input_image):
        """Encodes image and matches dimension to T5 embeddings.
        
        """
        return get_vit_features(self.vit, input_image)

    def forward(
        self,
        use_t5_encoder:bool, 
        input_image=None,
        input_embeds=None,
        labels=None,
        decoder_input_ids=None,
        past_key_values=None
        ):
        """Teacher forcing traning.
        For T5 details, please refer to
        `https://github.com/huggingface/transformers/blob/504ff7bb1234991eb07595c123b264a8a1064bd3/src/transformers/modeling_t5.py#L1136`
        """
        # Useful for generation, encoding image on first step only
        input_embeds = input_embeds if input_embeds is not None else self.encode_image(input_image)

        # Obs.: input_embeds are only used on encoder if encoder_outputs is None
        output = self.t5(
            encoder_outputs=None if self.hparams.use_t5_encoder else (input_embeds,),
            inputs_embeds=input_embeds,
            labels = labels,
            return_dict=True,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values
        )
        return output
    
    @torch.no_grad()
    def greedy_generate(self, input_image, max_length, use_t5_encoder:bool):
        """Greedy token generation.

        """
        # T5: 0
        eos_token_id = self.tokenizer.eos_token_id
        # T5: 1 (same as padding)
        decoder_start_token_id = self.t5.decoder.config.decoder_start_token_id

        input_ids = torch.full(
            (input_image.shape[0], 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=input_image.device
        )

        # First pass, outside loop
        image_features = self.encode_image(input_image)
        if use_t5_encoder:
            input_embeds = self.t5.get_encoder()(inputs_embeds=image_features)[0]
        else:
            input_embeds = image_features


        past = None
        cur_len = 1
        
        while cur_len < max_length:
            outputs = self(
                input_embeds=input_embeds,
                use_t5_encoder=False, # possible t5 encoder use already done
                decoder_input_ids=input_ids,
                # past_key_values=past
            )
            next_token_logits = outputs.logits[:, -1, :]

            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Avoids generation restarting after the first eos
            next_token[input_ids.eq(eos_token_id).any(-1)] = eos_token_id

            cur_len = cur_len + 1
            input_ids = torch.cat([input_ids,next_token.unsqueeze(-1)], dim=-1)

            # Check if output is end of senquence for all batches
            if torch.eq(next_token, eos_token_id).all():
                break
            
            # if model has past, then set the past variable to speed up
            # decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

        return input_ids
    
    def decode_token_ids(self, token_ids):
        """Decodifica tokens id e transforma em texto
        """
        decoded_text = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return decoded_text
    # --------------------------------------------------------------------------
    # Daqui pra baixo era pra estar sozinho em outra classe, mas não consegui
    # fazer
    # --------------------------------------------------------------------------

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate, eps=1e-08
            )
        return optimizer

    def _base_eval_step(self, batch):
        """Base function for eval/test steps.
        """
        true_captions = batch['raw_captions']

        generated_tokens = self.greedy_generate(
            input_image=batch['transformed_image'],
            max_length=self.hparams.max_tokens_captions_gen,
            use_t5_encoder=self.hparams.use_t5_encoder
        )
        generated_text = self.decode_token_ids(generated_tokens)
        rets = {'trues': true_captions,'preds': generated_text}
        return rets
    
    def _base_eval_epoch_end(self, outputs, prefix):
        """Base function for eval/test epoch ends.

            The following metrics are calculated:
            - BLEU: BLEU score, bleu-1, ... bleu-4
        """
        trues = sum([x['trues'] for x in outputs], [])
        preds = sum([x['preds'] for x in outputs], [])

        # Bleu score
        # bleu = sacrebleu.corpus_bleu(preds, [trues])
        bleu = sacrebleu.corpus_bleu(preds, trues)

        #         Some random examples
        idx_sample = random.choice(range(len(preds)))
        sample_trues = trues[idx_sample]
        sample_preds = preds[idx_sample]
        # sample_image = 
        print(
            80 * "-",
            f"\nSample predictions epoch {self.current_epoch} '{prefix}':",
            f"\nTrues:\n {sample_trues}",
            f"\nPreds:\n {sample_preds}"
        )
        log_dict = {
            f"{prefix}_bleu_score": bleu.score,
            f"{prefix}_bleu-1": bleu.precisions[0],
            f"{prefix}_bleu-2": bleu.precisions[1],
            f"{prefix}_bleu-3": bleu.precisions[2],
            f"{prefix}_bleu-4": bleu.precisions[3]
        }
        return log_dict
        
    def training_step(self, batch, batch_idx):
        labels = batch['tokenized_target_caption']
        loss = self(
            use_t5_encoder=self.hparams.use_t5_encoder, 
            input_image=batch['transformed_image'],
            labels=labels.masked_fill(labels == 0, -100),
            )['loss']

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):        
        output = self._base_eval_step(batch)
        return output

    def test_step(self, batch, batch_idx):        
        output = self._base_eval_step(batch)
        return output

    def validation_epoch_end(self, outputs):
        output = self._base_eval_epoch_end(outputs, 'val')
        for k,v in output.items():
            self.log(k, v, prog_bar=True)

    def test_epoch_end(self,outputs):
        output = self._base_eval_epoch_end(outputs, 'test')
        for k,v in output.items():
            self.log(k, v, prog_bar=True)
        return output