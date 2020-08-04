#just skeleton code

gradient = 2 * input_data * error
print(gradient)

weights_updated = weights - learning_rate * gradient

preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target

print(error_updated)
