There are 30 satellite pictures of houses and 25 corresponding labels that indicate the roofs. Take those 25 data points and train a neural network on them - you are completely free about the architecture and are of course allowed to use any predefined version of networks, however, you should be able to explain what you are doing - in terms of code as well as in terms of why certain steps are good choices. The preferred language is Python, but you can also use other languages. Please evaluate your network on the 5 remaining test images by making predictions of the roofs - send us the predictions and ideally some comments on what you have been doing. 

Don't hesitate to ask questions at any time. It would be great if you could provide us with your results (incl. code) within 3 weeks. A GPU is not necessary, but of course speeds up the training - hence the hint that among others Google Colab offers a free use of GPUs. Good luck!


#### Loading and training the model without augmented images
# Define the model
unet = UNet()

# Compile the model and train it
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Getting images
imgs = img('normal')

# Train the model
h_unet = unet.fit(imgs[0],imgs[1],epochs=30, batch_size=5, validation_data = (imgs[2],imgs[3]))


# Plotting the accuracy and loss for train and validation data
acc_loss_plot(h_unet)

# Plotting the ROC curve
tprs,fprs = roc_curve(unet)

# Predictions on the test set
test_predictions(unet,tprs,fprs)