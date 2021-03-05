from .core.batch_to_ids import TextTransformer, batch_to_ids
if not hasattr(batch_to_ids, 'text_transformer'):
    batch_to_ids.text_transformer = TextTransformer()


char_embedder_params = {
    'char_embedding_dim': 64,
    'max_characters_per_token': batch_to_ids.text_transformer.max_characters_per_token,
    'n_characters': batch_to_ids.text_transformer.max_char_idx + 1,
    'cnn_options': [
        [1, 128],
        [2, 128],
        [3, 256],
        [4, 512],
        [5, 512],
        [6, 1024],
        [7, 2048]],
    'n_highway': 2,
    'output_dim': 512}

batch_size = 32

sru_model_params = {
    'n_layers': 8,
    'output_dim': 512
}

lstm_model_params = {
    'n_layers': 2,
    'output_dim': 512
}
