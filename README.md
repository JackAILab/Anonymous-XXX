# Anonymous-XXX

This is a work in the field of AIGC that introduces FaceParsing information and FaceID information into the Diffusion model. Previous work mainly focused on overall ID preservation, even though fine-grained ID preservation models such as InstantID have recently been proposed, the injection of facial ID features will be fixed. In order to achieve more flexible consistency maintenance of fine-grained IDs for facial features, a batch of 50000 multimodal fine-grained ID datasets was reconstructed for training the proposed FacialEncoder model, which can support common functions such as personalized photos, gender/age changes, and identity confusion.

At the same time, we have defined a unified measurement benchmark FGIS for Fine-Grained Identity Preservice, covering several common facial personalized character scenes and characters, and constructed a fine-grained ID preservation model baseline.

Finally, a large number of experiments were conducted in this article, and ConsistentID achieved the effect of SOTA in facial personalization task processing. It was verified that ConsistentID can improve ID consistency and even modify facial features by selecting finer-grained prompts, which opens up a direction for future research on Fine-Grained facial personalization.


