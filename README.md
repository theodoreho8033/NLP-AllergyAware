# NLP-AllergyAware

This project presents a comprehensive exploration of recipe generation using various language models, focusing on developing robust
evaluation metrics and allergen-aware recipe
generation. We conducted extensive experiments with multiple model architectures, ranging from T5-small (Raffel et al., 2023) and
SmolLM-135M (Allal et al., 2024) to Phi-2 (Research, 2023), implementing both traditional
NLP metrics and custom domain-specific evaluation metrics. 

Our novel evaluation framework incorporates recipe-specific metrics for
assessing content quality and introduces an approach to allergen substitution. The results
indicate that, while larger models generally
perform better on standard metrics, the relationship between model size and recipe quality is more nuanced when considering domainspecific metrics. We find that SmolLM-360M
and SmolLM-1.7B demonstrate comparable
performance despite their size difference, while
Phi-2 shows limitations in recipe generation
despite its larger parameter count. 

Our comprehensive evaluation framework and allergen substitution system provide valuable insights for
future work in recipe generation and broader
NLG tasks that require domain expertise and
safety considerations.




