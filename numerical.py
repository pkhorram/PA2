print(model.layers[0].w.shape)
print(len(model.layers))

tests = [(-1, 'b', (0,0)), (-3, 'b', (0,0)), (-1, 'w', (0,0)), (-1, 'w', (1,0)), (0, 'w', (0,0)), (0, 'w', (1,0))]
epsilon = 0.01

for test in tests:
    layer_idx = test[0]
    weight_type = test[1]
    weight_idx = test[2]
    
    seen_classes = set()
    xsamples = []
    ysamples = []
    for i in range(len(train_images)):
        xsample = train_images[i].reshape((-1,1))
        ysample = train_labels[i].reshape((-1,1))
        temp = np.argwhere(ysample)
        yclass = temp[0][0]
        if yclass not in seen_classes:
            ysamples.append(ysample)
            xsamples.append(xsample)
            seen_classes.add(yclass)
        if len(seen_classes) == 10:
            break
    
#     print(ysample.shape)
    
    for i in range(len(ysamples)):
        xsample = xsamples[i]
        ysample = ysamples[i]
        
        print('For sample # ', i + 1)
    
        model1 = copy.deepcopy(model)
        model1.forward(xsample, ysample)
        model1.backward()
        if weight_type == 'w':
            grad1 = model1.layers[layer_idx].v_w[weight_idx]
        elif weight_type == 'b':
            grad1 = model1.layers[layer_idx].v_b[weight_idx]
        
        model0 = copy.deepcopy(model)

        if weight_type == 'w':
            model0.layers[layer_idx].w += epsilon
            output, loss1 = model0.forward(xsample, ysample)
            model0.layers[layer_idx].w -= 2*epsilon
            output, loss2 = model0.forward(xsample, ysample)
            slope = (loss1 - loss2) / (2*epsilon)
        else:
            model0.layers[layer_idx].w += epsilon
            output, loss1 = model0.forward(xsample, ysample)
            model0.layers[layer_idx].w -= 2*epsilon
            output, loss2 = model0.forward(xsample, ysample)
            slope = (loss1 - loss2) / (2*epsilon)

        print('grad1 - slope: ', grad1 - slope)
        print('eps square: ', epsilon**2)
    
#     break