"""Small 140M Parcae model configuration with Multi-head Latent Attention."""

from parcae_lm.registry import DEFAULT_HF_ORG

configs = [
    dict(
        name="parcae-140m-mla",
        hf_config=dict(org=DEFAULT_HF_ORG, name="parcae-140m-mla"),
        # Context
        block_size=2048,
        vocab_size=32768,
        padding_multiple=64,
        n_embd=768,
        num_attention_heads=6,
        num_key_value_heads=6,
        intermediate_size=3072,  # 4x n_embd
        bias=False,
        tie_embeddings=True,
        prelude_norm=True,
        # Model class
        architecture_class_name="Parcae",
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU2",
        qk_norm=False,
        logit_softcap=None,
        # Multi-head Latent Attention
        attention_class_name="MLACausalSelfAttention",
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        # Initialization
        init_strategy="scaled-zero",
        init_orthogonal=True,
        # Recurrent settings - diagonal injection
        injection_type="diagonal",
        state_init="like-init",
        recurrent_embedding_dimension=768,
        recurrent_intermediation_embedding_dimension=3072,
        n_layers_in_recurrent_block=2,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        mean_recurrence=8,
        sampling_scheme="poisson-truncated-full",
        mean_backprop_depth=4,
        recurrent_iteration_method="per-sequence",
    ),
]
