# Validation and test comparison

Saved validation predictions were evaluated again with the exact Phase 5C content-order policy used for test. No validation inference was repeated and no historical report was overwritten. These references use V3 for final_C/final_F and therefore must not be substituted for historical V2 C/F.

| Condition | Validation mAP50–95 | Test mAP50–95 | Test−val pp | Val fracture AP50–95 | Test fracture AP50–95 |
| --- | --- | --- | --- | --- | --- |
| final_A | 0.33382020 | 0.31708115 | -1.6739 | 0.51968388 | 0.50524317 |
| final_B | 0.29317763 | 0.28512014 | -0.8057 | 0.49159750 | 0.47774610 |
| final_C | 0.33382861 | 0.31709538 | -1.6733 | 0.51974347 | 0.50535441 |
| final_D | 0.33758845 | 0.33025835 | -0.7330 | 0.51201845 | 0.49970648 |
| final_E | 0.28886177 | 0.28358515 | -0.5277 | 0.47584538 | 0.47095515 |
| final_F | 0.33761513 | 0.33029334 | -0.7322 | 0.51221780 | 0.49989725 |

| Contrast | Validation delta pp | Test delta pp | Val 95% pp | Test 95% pp |
| --- | --- | --- | --- | --- |
| D−A (training) | +0.3768 | +1.3177 | [-3.1509, +2.4861] (n=492) | [+0.2208, +3.0145] (n=492) |
| E−B (training) | -0.4316 | -0.1535 | [-4.3220, +1.6992] (n=492) | [-1.2205, +1.7359] (n=492) |
| F−C (training) | +0.3787 | +1.3198 | [-3.1486, +2.4848] (n=492) | [+0.2209, +3.0192] (n=492) |
| B−A (SAHI) | -4.0643 | -3.1961 | [-4.8272, -3.1160] (n=492) | [-4.3818, -2.0918] (n=492) |
| E−D (SAHI) | -4.8727 | -4.6673 | [-5.8005, -3.7018] (n=492) | [-5.6950, -3.3678] (n=492) |
| C−A (V3, primary) | +0.0008 | +0.0014 | [-0.0028, +0.0058] (n=492) | [-0.0046, +0.0076] (n=492) |
| F−D (V3) | +0.0027 | +0.0035 | [-0.0032, +0.0122] (n=492) | [-0.0043, +0.0124] (n=492) |
| C−B (V3−SAHI) | +4.0651 | +3.1975 | [+3.1162, +4.8280] (n=492) | [+2.0923, +4.3825] (n=492) |
| F−E (V3−SAHI) | +4.8753 | +4.6708 | [+3.7021, +5.8038] (n=492) | [+3.3716, +5.6986] (n=492) |
| (E−D)−(B−A) | -0.8084 | -1.4712 | [-1.5972, +0.0470] (n=492) | [-2.3467, -0.3667] (n=492) |
| (F−D)−(C−A) | +0.0018 | +0.0021 | [-0.0060, +0.0119] (n=492) | [-0.0091, +0.0126] (n=492) |

Both cohorts use 500 patient-paired samples/seed314159, with cohort-specific patient draws and class-presence counts. Intervals compare methods within each partition; they are not intervals for the test-minus-validation difference. Signs and intervals show which observed patterns persist; a matching sign alone is not statistical confirmation. Different patient/class composition and sparse findings can shift scores. Do not interpret a cross-partition difference as a causal domain shift.

The main inference patterns persist: SAHI lowers overall/fracture AP on both partitions, while V3 has tiny positive overall point changes with intervals spanning zero. Rarity-only training has positive overall point changes under standard and V3 inference on both partitions; its test intervals exclude zero, while validation intervals do not. This does not test whether the training effect itself changed between partitions. Rarity-only fracture AP remains lower at the point estimate. All six test mAP50–95 values are below their comparable validation references, by 0.5277–1.6739 pp.

The main scientific interpretation is in [FINAL_SCIENTIFIC_ASSESSMENT.md](FINAL_SCIENTIFIC_ASSESSMENT.md). Neither this comparison nor the test results are used to change the fixed checkpoints, V3, thresholds or primary configuration.
