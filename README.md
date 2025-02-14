# PIMPNet3D
We present Patch-based Intuitive Multimodal Prototypes Network (PIMPNet), an interpretable multimodal model for 3D images and demographics.
We applied PIMPNet to the binary classification of Alzheimer's Disease from 3D structural Magnetic Resonance Imaging (sMRI, T1-MRI) and patient’s age.

Classes (clinical cognitive decline level):

- Cognitively Normal (CN)
- Alzheimer's Disease (AD)


**arXiv preprint**: [_"Patch-based Intuitive Multimodal Prototypes Network (PIMPNet) for Alzheimer’s Disease classification"_](https://arxiv.org/pdf/2407.14277)

Presented as _late-breaking work_ during the [The 2nd World Conference on eXplainable Artificial Intelligence](https://xaiworldconference.com/2024/the-conference/) in July 2024.

![Overview of PIMPNet](https://github.com/desantilisa/PIMPNet3D/blob/main/pimpnet_poster.png)   

Images, ages and labels (cognitive decline level) were collected from the Alzheimer's Disease Neuroimaging Initiative (ADNI) https://adni.loni.usc.edu (data publicity available under request).

Brain atlas (CerebrA) downloaded from https://nist.mni.mcgill.ca/cerebra/.

Codes adapted from the original [PIPNet](https://github.com/M-Nauta/PIPNet/tree/main)

Training a PIMPNet: main_train_pimpnet.py

Test a trained PIMPNet: main_test_pimpnet.py

Link to the weights of trained PIMPNet(s) available in "models" folder.


