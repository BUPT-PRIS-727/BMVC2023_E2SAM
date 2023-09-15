# E2SAM: A Pipeline for Efficiently Extending SAM’s Capability on Cross-Modality Data via Knowledge Inheritance
Code release for "E2SAM" (BMVC 2023）. The brief introduction of our proposed method is shown as:
<img width="750" alt="image" src="https://github.com/BUPT-PRIS-727/BMVC2023_E2SAM/assets/86039485/7f4cd94d-5363-46e3-b020-4ba2cef82ac2">

**Abstract**: Segment Anything Model (SAM) has achieved brilliant results on many segmentation datasets due to its strong segmentation capability with visual-grouping perception. However, the limitation of the three-channel input means that it is difficult to apply directly to cross-modality data.
Therefore, this paper proposes a pipeline called $E^{2}SAM$ with the knowledge inheritance stage and downstream fine-tuning stage step by step that can efficiently inherit the capabilities of SAM and extend to both cross-modality data and relevant task-specific application.
In order to enable the feature alignment of varying single-modality to cross-modality data, an auxiliary branch with a channel selector and a merge module is designed in the first stage. It is worth noting that we do not need a large amount of additional annotated training data during our pipeline.
Furthermore, the strengths of the proposed method are discussed in detail through experiments on generalization performance and resistance to size changes. The experimental results and visualizations on the SFDD-H8 and SHIFT datasets demonstrate the effectiveness of our proposed methods compared to other methods such as random initialization and SAM-based fine-tuning.

# Requirements
mmengine>=2.0

pytorch>=2.0

# Usage
1. Download dataset from 

   [SHIFT ]: https://www.vis.xyz/shift/download/

   | Images (1 fps) | train/val |
   | --------------------------- | ----- |
   | RGB Image Front             | 12.6G / 2.0G |
   | Semantic Segmentation Front | 3.9G / 0.667G |
   | Depth Maps Front            | 68.4G / 11.4G |
   

2. Preceed data

   Release later

3. run

   ```
   # Training:
   python runner_shift_4_sam.py
   # Inference:
   python infer.py
   ```


# Citation

If you find this paper useful in your research, please consider citing:


# Contact
Thanks for your attention! If you have any suggestion or question, you can leave a message here or contact us directly:
- susundingkai@bupt.edu.cn
- wuming@bupt.edu.cn
