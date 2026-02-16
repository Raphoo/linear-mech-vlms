# Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models

This repository contains the implementation of the paper ["Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models"](https://arxiv.org/pdf/2601.12626) **(ICLR 2026)**. In this work, we show that VLMs such as LLaVA, LLaMA, Qwen, Gemma, and InternVL perform spatial reasoning via _spatial IDs_. We perform rigorous causal interventions and steering analyses to illustrate the IDs' function.

If you find this repository useful, please consider citing our work!

```bibtex
@article{kang2026spatiotemporalreasoning,
      title={Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models}, 
      author={Raphi Kang and Hongqiao Chen and Georgia Gkioxari and Pietro Perona},
      journal={ICLR},
      year={2026},
      url={https://arxiv.org/abs/2601.12626}, 
}
```



## Experiments

| Figure | Description | Directory |
|--------|-------------|-----------|
| **Figure 2** | Adversarial Steering - Belief Swapping | [`adversarial_steering/`](adversarial_steering/) |
| **Figure 4** | Mirror and Attribute Swapping | [`mirror_attr_swapping/`](mirror_attr_swapping/) |
| **Figure 5** | Spatial IDs in a Grid | [`spatial_id_derivation/`](spatial_id_derivation/) |
| **Figure 6** | Arbitrary Spatial Steering | [`arbitrary_steering/`](arbitrary_steering/) |
| **Figure 7** | Depth Diagnosis | [`depth_diagnosis/`](depth_diagnosis/) |
| **Figure 8A** | Ground Truth Spatial ID Deviation | [`ground_truth_deviation/`](ground_truth_deviation/) |
| **Figure 8B** | Image Masking (D-RISE) Sensitivity | [`ground_truth_deviation/`](ground_truth_deviation/) |
| **Figure 9** | Accuracy vs. Steerability | [`adversarial_steering/`](adversarial_steering/) |
| **Figure 10** | Temporal IDs in Video Models | [`temporal_models/`](temporal_models/) |

Please refer to the README of each experiment for detailed instructions.

## Environment

All experiments use the conda environment defined in `environment.yml` with the exception of `temporal_models`. For temporal models, use `temporal_models/environment.yml`.
