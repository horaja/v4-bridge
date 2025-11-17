# Core Hypothesis

Aligning intermediate layers of CNN/ResNet/ViT to macaque V4 visual cortex representations can improve domain generalization, particularly enabling color-trained networks to better recognize line drawings.

# Background Context

1. CLIP and other vision models share geometric similarities in representations to primate visual cortex.
2. V4 is known to have 2 functional regions
	1. Shape-biased region
		1. Sensitive to 3D shapes
	2. Texture-biased region

# Proposed Methods

1. Geometric Alignment
	1. Align *(manifolds of)* intermediate layer representations to V4 geometry using methods like CCA, CKA, optimal transport, RSA, etc.
		1. Explore Lie Group Actions - [[Lie Group Model]]
	2. Fine-tune weights prior to the intermediate layer
	3. Test impact on neural response prediction and domain generalization
2. Neural Response Prediction Alignment
	1. Use linear regression *(or alternative)* from intermediate layer activations to predict individual V4 neuron responses
	2. Fine-tune through backpropagation to optimize prediction accuracy
	3. Assess whether this implicitly produces geometric alignment
3. Combined Approach
	1. Apply both geometric and neural response prediction alignment
	2. Evaluate cumulative effects on classification performance
		![[V4 Experiments Model]]
## Some Considerations
1. Evaluate benefit of biological learning vs general feature learning using standard transfer learning methods
2. Critical Stimulus Selection

# Expected Outcomes
1. Improved line drawing recognition (domain generalization)
2. Better photo classification performance
3. Enhanced compositional understanding —*Separate Task to test on?*
4. More human-like visual processing through incorporation of V4's 3D part sensitivity

# Research Impact
1. How biological visual processing principles can improve artificial vision systems
2. The role of intermediate representations in domain generalization
3. The relationship between geometric and functional alignment in neural networks