# Linear probe of photo features (avocado, val split)

Ridge from one visit's features to its stage and to days until spoiled; train split fit, val split scored.

| cache | dim | train visits | val visits | stage MAE | stage acc. | days-left MAE |
|---|---|---|---|---|---|---|
| avocado_resnet18 | 512 | 4394 | 969 | 0.362 | 0.714 | 2.910 |
| avocado_dinov2_vitb14 | 768 | 4394 | 969 | 0.348 | 0.759 | 2.963 |
