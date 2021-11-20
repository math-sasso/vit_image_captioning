class MyDataset(Dataset):
    """Pytorch dataset para treinamento da rede

        h5_images_path (str): Arquivo hdf5 com as imagens
        json_captions_path (str): Json com as captions das imagens (5 por imagem)
        transforms (callable): pré-processamento de cada imagem
        return_dict (bool): se devemos retornar os resultados como um dict
        tokenizer: tokenizador a ser usado
        max_tokens_captions: tamanho máximo de tokens
        captions_per_image: número de legendas por imagem
    """

    def __init__(self,h5_images_path: str,json_captions_path: str,tokenizer,
                 max_tokens_captions,transforms=None, captions_per_image=5):
        self.h5_images_path = h5_images_path
        self.json_captions_path = json_captions_path
        
        self.caption_data = json.load(open(json_captions_path, 'r'))
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_tokens_captions = max_tokens_captions
        self.captions_per_image = captions_per_image
    
    def __len__(self):
        return len(self.caption_data)

    def __transforms(self,image):

    def _tokenize(self, input_text):
        tokenized = self.tokenizer(
            input_text, return_tensors='pt',padding='max_length',
            max_length=self.max_tokens_captions,pad_to_max_length=True,
            return_attention_mask=False, truncation=True,
            add_special_tokens=True
            )
        return tokenized['input_ids'].squeeze()

    def __getitem__(self, idx):
        raw_captions = self.caption_data[idx]
        
        image_data = h5py.File(self.h5_images_path, 'r')['images']
        
        num_captions = len(raw_captions)
        assert num_captions == self.captions_per_image

        # Random choice of one out of 5 captions
        chosen_target_caption = raw_captions[random.choice(range(num_captions))]
        tokenized_target_caption = self._tokenize(chosen_target_caption)
        
        image = torch.from_numpy(image_data[idx]).to(torch.uint8) # (C, H, W)
        transformed_image = image
        
        if self.transforms is not None:
            transformed_image = self.transforms(image)
        

        rets = {
            'original_image': image,
            'transformed_image': transformed_image,
            'raw_captions': raw_captions,
            'tokenized_target_caption': tokenized_target_caption
            }
        return rets
    
    @staticmethod
    def collate_fn(batch):
        original_image = torch.stack(
            [item['original_image']
             for item in batch], dim=0)
        
        transformed_image = torch.stack(
            [item['transformed_image'] for item in batch], dim=0)
        
        # List of lists: (batch_size * captions_per_image)
        raw_captions = [item['raw_captions'] for item in batch]
        # raw_captions = list(zip(*[item['raw_captions'] for item in batch]))

        tokenized_target_caption = torch.stack(
            [item['tokenized_target_caption'] for item in batch], dim=0)
        
        rets = {
            'original_image': original_image,
            'transformed_image': transformed_image,
            'raw_captions': raw_captions,
            'tokenized_target_caption': tokenized_target_caption
            }
        return rets