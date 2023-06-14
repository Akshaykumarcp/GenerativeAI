# GenerativeAI
GenerativeAI Repo

Applications of GANs 
- text-to-image translation
  - Creating Art
  - style transfer
  - photo colorization
- image-to-image translation
  - Image Super-Resolution
  - Creating Art
  - transforming photos from summer to winter or day to night

GANs is successful ability to model high-dimensional data, handle missing data, and the capacity of GANs to provide multi-modal outputs or multiple plausible answers.

GANS
| Variant  | Paper | Year | Description
| ------------- | ------------- | ------------- | ------------- |
| Generative Adversarial Networks (GAN)  | [link](https://arxiv.org/abs/1406.2661)  | 10 Jun 2014  | The two models, the generator and discriminator, are trained together. The generator generates a batch of samples, and these, along with real examples from the domain, are provided to the discriminator and classified as real or fake. The discriminator is then updated to get better at discriminating real and fake samples in the next round, and importantly, the generator is updated based on how well, or not, the generated samples fooled the discriminator. |
| Conditional Generative Adversarial Nets  | [link](https://arxiv.org/abs/1411.1784) |  6 Nov 2014  | The generative model can be trained to generate new examples from the input domain, where the input, the random vector from the latent space, is provided with (conditioned by) some additional input. |
| NIPS 2016 Tutorial: Generative Adversarial Networks  | [link](https://arxiv.org/abs/1701.00160) |  31 Dec 2016 | - |
