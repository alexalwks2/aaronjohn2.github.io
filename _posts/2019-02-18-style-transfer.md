---
layout: post
comments: true
title:  "Performing Style Transfer Using a Convolutional Neural Network"
date:  2019-02-18 10:34:41
categories: CNN, Convolutional Neural Network, style transfer, PyTorch, VGG 19
---

CNN's (Convolutional Neural Networks) are some of the most powerful networks for image classification and analysis. CNN's process visual information in a feedforward manner; passing a input image through a collection of image filters, which extract certain features from the input image. It turns out that these feature level representations are not only useful for classification, but for image construction as well. These representations are the basis for applications like style transfer and [deep dream,](https://deepdreamgenerator.com/) which compose images based on CNN layer activations and extracted features.

In this documentation I will discuss the steps to take in implementing the style transfer algorithm. Style transfer allows you to apply the style of one image to another image of your choice. For example, here I have chosen a Salvador Dali painting for my style image and I chose a rowing picture of myself for my Content image in order to produce this final style transfer image as shown below.
    <div class="imgcap">
    <img src="/assets/bass/memory.jpg" height="400" width="600" class="center">
    <img src="/assets/bass/rowing_style_change.png">
    </div>
    
The key to this technique is using a trained CNN to separate the content from the style of an image. If you can do this, then you can merge the content of one image, with the style of another and create something entirely different.

**How Style and Content Can Be Separated**

When a CNN is trained to classify images, its convolutional layers learn to extract more and more complex features from the given image. Intermittently maxpooling layers will discard detailed spatial information; information that is increasingly irrelevant to the task of classification. The effect of this is that as we go deeper into a CNN, the input image is transformed into feature maps (convolutional layers) that increasingly care about the content of the image rather than any detail about the texture and color of pixels. Later layers of the network are even sometimes referred to as a content representation of an image.

In this way a trained CNN has already learned to represent the content of an image. But what about style?
Style can be thought of as traits that might be found in the brush strokes of a painting. Its texture, colors, curvature and so on. To perform style transfer we need to combine the content of one image with the style of anoyther. So how can we isolate only the style of an image?

To represent the style of an input image, a feature space designed to capture texture and color information is used. This space essentially looks at spacial correlations within a layer of a network. A correlation is a measure of the relationship between two or more variables. For example, you could look at the features extracted in the first convolutional layer, which has some depth. The depth corresponds to the number of feature maps in that layer. For each feature map we can measure how strongly its detected features relate to the other features maps in that layer. Is a certain color detected in one map similar to a color in another map. What about the differences between detected edges, corners and so on. See which colors and shapes in a set of feature maps are related and which are not. Say we detect that mini feature maps in the first convolutional layer have similar orange edge features. If there are common colors and shapes among the feature maps, then this can be thought of as part of that image's style.

So, the similarities and differences between features in a layer should give us some information about the texture and color found in an image. But at the same time, it should leave out information about the actual arrangement and identity of different objects in that image. This shows that content and style can be separate components of an image. Lets look at this in a complete style transfer example.

Style transfer will look at two different images, we often call these the style image and the content image. Using a trained CNN, style transfer finds the style of one image and the content of the other. Finally, it tries to merge the two to create a new third image. In this newly created image, the objects and their arrangement are taken from the content image, and the colors and textures are taken from the style image. Here's an example of an [image of a dog, the content image, being combined with an image of a blue painting.](https://news.developer.nvidia.com/wp-content/uploads/2018/08/Linear-Style-Transfer-featured.png) Effectively, style transfer creates a new image that keeps the dog content, but renders it with the colors, the print texture, and the style of the blue painting. This is the theory behind how style transfer works.

**How to Extract Features From Different Layers of a Trained Model**

Here we will extract features from different layers of a trained model and use them to combine style and content of two different images. In the code example we will recreate a style transfer method that is outlined in the paper [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). In this [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), style transfer uses the features found in the 19 layer VGG network (VGG 19). This network accepts a color image as input and passes it through a series of convolutional and pooling layers. Followed, finally by three fully connected layers that classify the passed-in image. In-between the five pooling layers, there are stacks of two or four convolutional layers. The depth of these layers is standard within each stack, but increases after each pooling layer.

In the paper, the content representation for an image is taken as the output from the fourth convolutional stack, conv4_2. As we form our new target image, we'll compare it's content representation with that of our content image. These 2 representations should be close to the same even as our target image changes it's style. To formalize this comparison, we'll define a content loss, a loss that calculates the difference between the content and target image representations, which I'll call Cr and Tr respectively. In this case, we calculate the mean squared difference between the two representations.

The content loss measures how far away these two representations are from one another.
Content loss = 1/2(sigma(Tr - Cr)^2)

As we try to create the best target image, are aim will be to minimize this loss. This is similar to how we used loss and optimization to determine the weights of a CNN during training. But this time are aim is not to minimize classification error. In fact, we're not training the CNN at all. Rather our goal is to change only the target image, updating its appearance until it's content representation matches that of our content image. So, we're not using the VGG 19 network in a traditional sense, we're not training it to produce a specific output. But we are using it as a feature extractor, and using back propagation to minimize a defined loss function between our target and content images. In fact, we'll have to define a loss function between our target and style images, in order to produce an image with our desired style.

**How to Represent the Style of an Image**

To make sure that our target image has the same content as our content image, we formalize the idea of a content loss, that compares the content representations of the two images. Next, we want to do the same thing for the style representations of our target image and style image. The style representation of an image relies on looking at correlations between the features in individual layers of the VGG 19 network, in other words looking at how similar the features in a single layer are. Similarities will include the general colors and textures found in that layer. We typically find the similarities between features in multiple layers in the network.

By including the correlations between multiple layers of different sizes, we can obtain a multiscale style representation of the input image; one that captures large and small style features. The style representation is calculated as an image passes through the network at the first convolutional layer in all five stacks, conv1_1, conv2_1, up to conv5_1. Here, the correlations at each layer are given by a *Gram Matrix*. The matrix is a result of a couple of operations, and its easiest to see in a simple example.

Say, we start off with a four by four image, and we convolve it with eight different image filters to create a convolutional layer. This layer will be four by four in height and width, and eight in depth. Thinking about the style representation for this layer, we can say that this layer has eight feature maps that we want to find the relationships between. The first step in calculating the Gram Matrix, will be to vectorize the values in this layer. This is probably very similar to what you've seen before in the case of vectorizing an image so that it can be seen by an MLP.

The first row of four values in the feature map, will become the first four values in a vector with length 16. The last row will be the last four values in that vector. By flattening the XY dimensions of the feature maps, we're converting a 3D convolutional layer into a 2D matrix of values. The next step is to multiply this matrix by its transpose. Essentially, multiplying the features in each map to get the Gram Matrix. This operation treats each value in the feature map as an individual sample, unrelated in space to other values. So, the resulted Gram Matrix contains non-localized information about the layer. Non-localized information, is information that would still be there even if an image was shuffled around in space.

For example, even if the content of a filtered image is not identifiable, you should still be able to see prominent colors, and textures of the style. Finally, we're left with the square, eight by eight G (Gram Matrix), whose values indicate the similarities between the layers. Hence, G, row four column two, will hold a value that indicates the similarity between the fourth and second feature maps in a layer. Importantly, the dimensions of this matrix are related only to the number of feature maps in the convolutional layer. It doesn't depend on the dimensions of the input image.

I should note that the Gram Matrix is just one mathematical way of representing the idea of shared in prominent styles. Style itself is an abstract idea, but the Gram Matrix is the most widely used in practice. Now that we've defined Gram Matrix as having information about the style of a given layer, next...

**Calculating a Style Loss** *(that Compares the Style of Our Target Image and Our Style Image)*

To calculate the style loss between a target and style image, we find the mean squared distance between the style and target image Gram Matrices, all five pairs that are computed at each layer in our predefined list, conv1_1 up to conv5_1. These lists I'll call "Sl" and "Tl", where 'a' is a constant that accounts for the number of values in each layer. We'll multiply these five calculated distances by some style weights W that we specify, and then add them up. (where sigma runs from i to some value)
Style Loss = a(sigma(w)(Tl - Sl)^2

The style weights are values that will give more or less weight to the calculated style loss at each of the five layers, thereby changing how much effect each layer style representation will have on our final target image. Again, we'll only be changing the Target image's style representations as we minimize this loss over some number of iterations. So, now we have the content loss, which tells us how close the content of our target image is to that of our content image and the style loss, which tells us how close our target is in style to our style image. We can now add these losses together to get the total loss, and then use typical back propagation and optimization to reduce this loss by iteratively changing the target image to match our desired content and style.

**Loss Weights**

We have values for the content and style loss, but since they are calculated differently, these values will be pretty different, and we want our target image to take both into account fairly equally. So, it's necessary to apply constant weigts, alpha and beta, to content and style losses respectively. Such that the total loss reflects in equal balance. In practice, this means multiplying the style loss by a much larger weight value than the content loss.

You'll often see this expressed as a ratio of the content and style weights, alpha over beta. In the paper [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) we see the effects of a bigger or smaller ratio. Here, is an [example](https://news.developer.nvidia.com/wp-content/uploads/2018/08/Linear-Style-Transfer-featured.png) of a content and style image. We can imagine that the content (dog) weight alpha is one, and that the style (blue painting) weight beta is 10. You can see that the target image is mostly content without much style as shown in the [example](https://news.developer.nvidia.com/wp-content/uploads/2018/08/Linear-Style-Transfer-featured.png). However, say if beta increases to 100 and alpha stays at 1 then we will be able to see more style in the generated image.

If beta is increased to 10^-4 this can be way to much, since most of the content is gone. Hence, the smaller the alpha-beta ratio, the more stylistic effect you will see. This makes intuitive sense because the smaller a ratio corresponds to a larger value for beta, the style weight. You may find that certain ratios work well for one image, but not another. These weights will be good values to change to get the exact kind of stylized effect that you want. Now that we know the theory and math behind using a pre-trained CNN to separate the content and style of an image, next we will see how to implement style transfer in PyTorch.

**Style Transfer with Deep Neural Networks Using PyTorch**

Here, we will go over an implementation of style transfer following the deatils outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) We're going to use a pre-trained VGG 19 net as a feature extractor. We can put individual images through this network, then at specific layers get the output, and calculate the content and style representations for an image. Basically, style transfer aims to create a new target image that tries to match the content of a given image and the style of a given style image.

1. Here's my [example](https://news.developer.nvidia.com/wp-content/uploads/2018/08/Linear-Style-Transfer-featured.png) of a dog and a blue painting form the target style image. With the code shown below, you'll be able to upload images of your own and really customize your own target style image. First, we will go ahead and load in our libraries.
    ```python
    # import resources
    %matplotlib inline

    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    import torch
    import torch.optim as optim
    from torchvision import transforms, models
    ```

2. Next, we will load in the pre-trained VGG 19 network that this implementation relies on. Using PyTorch's models we can load this network in by name and ask for it to be pre-trained. We actually just want to load in all the convolutional and pooling layers, which in this case are named as features, and this is unique to the VGG network. Here, we will load in a model and we will freeze any weights or parameters that we don't want to change. So, we are saving this pre-trained model using the VGG variable, then for every weight in this network we are setting requires_grad to False. This means that none of these weights will change. Hence, VGG becomes a kind of fixed feature extractor, which is just what we want for getting content and style features later.
    ```python
    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)
    ```

3. Next, we will check if a GPU device is available, and if it is, we will move our model to it. (It is recommended running this CNN on a GPU just to speed up the target image creation process.) Here, vgg.to(device) will print out the VGG model and all it's layers. (Once printed out on console, you will notice that the sequence of all layers is numbered)
    ```python
    # move the model to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg.to(device)
    ```

4. Next, we will continue loading the resources we need to implement style transfer. So, we have our trained VGG model and now we need to load in our content and style images. Below  we have a function which will transform any image into a normalized tensor. This will deal with jpg or png, and it will make sure that the size is reasonable for our purposes.
    ```python
    def load_image(img_path, max_size=400, shape=None):
        ''' Load in and transform an image, making sure the image
           is <= 400 pixels in the x-y dims.'''

        image = Image.open(img_path).convert('RGB')

        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape

        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)

        return image
    ```

5. Next, we will actually load in style and content images from our images directory. Here, we will reshape out style image into the same shape as the content image. This reshaping step is just going to make the math nicely lined up later on.
    ```python
    # load in content and style image
    content = load_image('images/rowing.jpg').to(device)
    # Resize style to match content, makes code easier
    style = load_image('images/memory.jpg', shape=content.shape[-2:]).to(device)
    ```

6. Next, we will also have a function to help us convert a normalized tensor image back into a numpy image for display. Here, I can show you the images that I choose. I chose a rowing picture of myself for my Content image and a Salvador Dali painting for my style image. Now, we have all the elements we need for style transfer.
    ```python
    # helper function for un-normalizing an image
    # and converting it from a Tensor image to a NumPy image for display
    def im_convert(tensor):
        """ Display a tensor as an image. """

        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image
    ```
    <div class="imgcap">
    <img src="/assets/bass/rowing.jpg">
    <img src="/assets/bass/memory.jpg">
    </div>

7. Next, we know we eventually have to pass our content and style images through our VGG network, and extract content and style features from particular layers. Below we have a get_features function which takes in an image and returns the outputs from layers that correspond to our content and style representations. This is going to be a list of features that are taken at particular layers in our VGG 19 model.
    ```python
    def get_features(image, model, layers=None):
        """ Run an image forward through a model and get the features for
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """

        ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
        ## Need the layers for the content and style representations of an image
        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',  ## content representation
                      '28': 'conv5_1'}

        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features
    ```
(Note - The descriptive dictionary maps our VGG 19 layers, that are currently numbered from 0 through 36, into names like conv1_1, conv2_1 and so on. If we look at the original paper, [Image Style Transfer Using Convolutional Neural Networks.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) and scroll down close to the bottom, we can see exactly which layers make up our style and content representations. In that paper we will notice that the content representation is taken as the output of layer conv4_2, and the style representations are going to be made of features from the first convolutional layer in all five stacks, conv1_1 up to conv5_1. So below we will map the number sequences that point to the VGG 19 layers as shown in step 3 to their more descriptive name. So 0 is conv1_1, which is easy enough. Next, after the maxpooling layer, we can see that we have conv2_1 at layer 5. Then, we will look for the next maxpooling layer and find that conv3_1 is at layer 10. We also want to get conv4_1 and con4_2 for our content representation. Lastly, we want to get conv5_1.)

8. Next, we need to pass a style image through the VGG model and extract the style features at the right layers which we've just specified. Then, once we get these style features at a specific layer, we'll have to compute the Gram Matrix. This function takes in a tensor, the output of a convolutional layer, and returns the Gram Matrix, the correlations of all the features in that layer.
    ```python
    def gram_matrix(tensor):
        """ Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """

        # get the batch_size, depth, height, and width of the Tensor
        _, d, h, w = tensor.size()

        # reshape so we're multiplying the features for each channel
        tensor = tensor.view(d, h * w)

        # calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())

        return gram
    ```
(Note - here, we take a look at the shape of the passed in tensor. This tensor should be four dimensional with a batch size, a depth, a height, and width. You'll need to reshape this so that the height and width are flattened, and then you can calculate the Gram Matrix by doing matrix multiplication. Hence, in summary the gram_matrix function takes in a tensor which will be the output of some convolutional layer. Then the first thing we do is take a look at its size. Each tensor is going to be four dimensional with a batch size, a depth, a height, and a width. We ignore the batch size at this point, because we're only interested in the depth, or number of feature maps, and the height and width. These dimensions then tell me all we need to know to then factorize this tensor. Next, we are reshaping this tensor so that it is now a 2D shape that has its spacial dimensions flattened. It retains the number of feature maps as the number of rows. Hence, its 'd' rows by 'h'  * 'w' columns. Finally, we calculate the Gram Matrix by matrix multiplying this tensor times its transpose. This effectively multiplies all the features and gets the correlations. Finally, we make sure to return that calculated matrix.)

9. Next, we will put all the pieces together where we have a get_features function and a gram_matrix function. Now, before we start to form our target image, we know we want to get the features from our content and style image; where these will remain the same throughout this process. In the below code, we call get_features on our content image, passing in our content image and the VGG model; we do the same thing for our style features as well, passing in our style image and the VGG model.
    ```python
    # get content and style features only once before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # create a third "target" image and prep it for change
    # it is a good idea to start of with the target as a copy of our *content* image
    # then iteratively change its style
    target = content.clone().requires_grad_(True).to(device)
    ```
(Note - here, we calculate all the Gram Matrices for each of our style layers, conv1_1 up to conv5_1. This looks at all of the layers in our style features and computes the Gram Matrix. This returns a dictionary where we can call style_grams with a given layer name and get the Gram Matrix for that layer. Then, we create a target image. We could start with a blank slate, but it turns out to be easiest to just start with a clone of the content image. This way, our image will not divert too far from my rowing image content and our plan will be to iterate and change this image to stylize it more and more later. Hence, in preparation for changing this target image, we're going to set requires_grad to True, and we will move it to a GPU if available.)

10. Now, we will talk about how to set and calculate our style and content losses for creating interesting target images. Here, we will define style and content losses, where alongside we will define loss weights. This way, we can iteratively update our target image.
    ```python
    # weights for each style layer
    # weighting earlier layers more will result in *larger* style artifacts
    # notice we are excluding `conv4_2` our content representation
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta
    ```
(Note - we can play around with the values for the style_weights giving more weight to earlier layers. Firstly, we will define our style weights for each of our individual style layers. Notice that conv4_2 is excluded here. Now, these are just weights that are going to give one set of style features more importance than another's. For example, I prefer to weigh earlier layers a little bit more. These features are often larger due to the spatial size of these feature maps. Whereas weighting later layers might emphasize more fine grain features. But then again, this is just a preference and up to the coder to customize. It is recommended to keep the values within the zero to one range. Then we have our alpha and beta, where we will descriptively name our content weight and our style weight. Earlier we discussed this as a ratio, that makes sure that style and content are equally important in the target image creation process. Because of how style loss is calculated, we basically want to give our style loss a much larger weight than the content loss. Here, its 1 * 10^6 and content loss is just one. Now, if beta is too large, you may see too much of a stylized effect, but these values are good starting points.)

11. Next, we enter the iteration loop. Here is where we will actually change our target image. Now, keep in mind this is not a training process, hence, it's arbitrary where we stop updating the target image. It is recommended to have at least 2000 iterations. But you may want to do more or less depending on your computing resources and desired effect. In this iteration loop, when calculating the content loss, we will mean square the difference between the target and content representations. We can get those representations by getting the features from our target image, and then comparing those features at a particular layer, in this case conv4_2, to the features at that layer for our content image. We're going to subtract these two representations and square that difference and calculate the mean. This will give us our content loss. Next, in this loop we do something similar for the style loss. Only this time we have to go through multiple layers for our multiple representations for style. Recall that each of our relevant layers was listed in our style_weights dictionary above.
    ```python
    # for displaying the target image, intermittently
    show_every = 400

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = 2000  # decide how many iterations to update your image (5000)

    for ii in range(1, steps+1):

        # get the features from your target image
        target_features = get_features(target, vgg)

        # the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # display intermediate images and print the loss
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.show()
    ```
(Note - first, we got a list of target images features using our get_features function. Then we define our content loss by looking at the target_features at the layer conv4_2 and the content features at conv4_2. So, we are comparing our content image and our target image content representations. I found the distance between the two and calculated the mean squared difference. Then we move on to style loss. For this one we look at every layer in our style_weights dictionary. For each of these layers we get the target_feature at that layer. For example, this would be what happens if our target image goes through our VGG 19 network and hits conv1_1. The output of our convolutional layer is then fed into our gram_matrix function. This gives us our target_gram matrix. Earlier, we calculated a dictionary of Gram Matrices for our style image. So, we get the Gram Matrix for our style image (style_gram) by accessing that by layer. Then for layer_state_loss, we calculate the mean squared difference between our style_gram and target_gram matrix. Again, this is for a particular layer and I weighted by the weights that I specified in our style_weights dictionary. So for example, for the first layer conv1_1 (step 10) is multiplied by the difference between the target and style gram matrices by one. Then we are adding that layer_style_loss to our accumulated style_loss and just dividing it by the size of that layer. This effectively normalizes our layer style_loss. Hence, by the end of the for loop, we have the value for the style loss at all five of our convolutional layers added up. Finally, we can compute the total_loss, which is just our content_loss and style_loss summed up and multiplied by their respective weights. Our content_loss is multiplied by one and style_loss is multiplied by 1e6. Here, I ran this loop for 2000 iterations, but I showed intermittently images every 400. The loss was also printed out with the image which seemed to be quite large and a difference could be seen in my rowing image right away.)

12. Then at the end of my 2000 iterations, I displayed my content, and my target image side-by-side. You can see that the target image still looks a lot like a rowing boat?? In fact, i think I could have stylized this even more. However, this target image has some of the brushstroke texture from the Salvador Dali painting which I used as my style image.
    ```python
    # display content and final, target image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(im_convert(content))
    ax2.imshow(im_convert(target))
    ```
    <div class="imgcap">
    <img src="/assets/bass/rowing_style_change.png">
    </div>

13. Now, go create your style transfer image using this CNN :)
