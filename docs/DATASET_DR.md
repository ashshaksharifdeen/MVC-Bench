# Diabetic-retinopathy dataset preparation
Dataset preparation follows [SPSD-ViT](https://github.com/Chumsy0725/SPSD-ViT). Cite SPSD-ViT and the original APTOS, EyePACS, Messidor, and Messidor-2 dataset publications or official challenge pages in any derived work.

Obtain each dataset from its official distributor or from the data bundle referenced by SPSD-ViT. 

`classnames.txt` must contain one class per line, in exactly the same index order used by the labels. Use disease-aware names that are consistent across all four domains. A typical five-grade DR mapping is:

```text
0 no diabetic retinopathy
1 mild diabetic retinopathy
2 moderate diabetic retinopathy
3 severe diabetic retinopathy
4 proliferative diabetic retinopathy
```

Verify the actual labels and wording used by the branch before generating results. A mismatch between label indices and class-name order invalidates both accuracy and calibration.

