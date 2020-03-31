# GAN-experiments

pix2pix, CycleGAN, StyleGAN2 etc - experiment repositoty

## [pix2pix](./pix2pix/README.md) 

### Results

Trained on the [FACADE Dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/) for about 150 epochs

Following samples show the generated images

![sample1](./samples/pix2pix/e_150_1.jpg) 

![sample2](./samples/pix2pix/e_150_2.jpg) 

![sample3](./samples/pix2pix/e_150_3.jpg)

On the shoes dataset

Left to right -> Generated image, input sketch, target image

![example1](./pix2pix/data/shoes_generation.png)
![example2](./pix2pix/data/shoes_generated_1.png)

I haven't optimized any hyper-parameters or trained enough since this is just for fun!

### Train
 
 Change the dataset directories in the dataloader inside teh script and call
 
```python pix2pix/pix2pix.py```

### @TODO 

- [ ] Add argument parser 
- [ ] Clean-up network configs


## CycleGAN results

@TODO
