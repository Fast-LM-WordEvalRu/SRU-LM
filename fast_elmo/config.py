from .core.batch_to_ids import TextTransformer, batch_to_ids
if not hasattr(batch_to_ids, 'text_transformer'):
    batch_to_ids.text_transformer = TextTransformer()


char_embedder_params = {
    'char_embedding_dim': 16,
    'max_characters_per_token': batch_to_ids.text_transformer.max_characters_per_token,
    'n_characters': batch_to_ids.text_transformer.max_char_idx + 1,
    'cnn_options': [
        [1, 32],
        [2, 32],
        [3, 64],
        [4, 128],
        [5, 256],
        [6, 512],
        [7, 1024]],
    'n_highway': 2,
    'output_dim': 512}

batch_size = 64

sru_model_params = {
    'backward': False,
    'n_sru_layers': 2,
    'output_dim': 512
}
