# Visual Similarity-Based Image Recommendation System Using CNN Autoencoder on Fashion MNIST

This project implements a **visual image recommendation system** using a **convolutional autoencoder** trained on the **Fashion MNIST** dataset. It recommends visually similar images by encoding them into embeddings and comparing similarity using **cosine distance**.

---

##  Features

- CNN-based autoencoder for unsupervised visual feature extraction  
- Recommendation using cosine similarity in embedding space  
- Interactive selection of item by index or label (e.g., "Dress")  
- Accepts new 28×28 grayscale images for querying  
- Option to recommend based on random sample from selected class

---

##  How It Works

1. **Train an autoencoder** on Fashion MNIST to learn compressed representations (embeddings).
2. **Extract embeddings** using the encoder part of the model.
3. For a given input image:
   - Encode it into its embedding.
   - Compute cosine similarity with embeddings from training set.
   - Show top visually similar images.

---

##  Dataset

**Fashion MNIST** contains 28×28 grayscale images from 10 fashion categories:
```
0: T-shirt/top      5: Sandal  
1: Trouser          6: Shirt  
2: Pullover         7: Sneaker  
3: Dress            8: Bag  
4: Coat             9: Ankle boot
```

---

## Project Structure

- **Autoencoder Definition**
  - Builds a convolutional autoencoder with encoder and decoder parts.
  - Trained on Fashion MNIST to learn compact visual embeddings.

- **Embedding Extraction**
  - Uses the encoder part of the model to extract feature vectors from training images.

- **Recommendation Functions**
  - `get_similar_items(index)`: Computes cosine similarity between the input image and all training embeddings.
  - `show_recommendations(index)`: Displays the query image and top visually similar results from the training set.
  - `recommend_for_new_image(img)`: Takes a new 28x28 grayscale image and shows similar images.

- **Label-Based Recommendation**
  - `show_one_per_class()`: Displays one example image per label to help user choose.
  - `get_random_index_for_label(label)`: Selects a random image index from a specified class.
  - `show_recommendations(query_index)`: Used again to show recommendations based on class selection.

- **Input Support**
  - Supports both direct image indexing and new user-supplied grayscale images(28 x 28).

---

##  Usage

### Train the Autoencoder
```python
autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test)
)
```

### Recommend Based on Index
```python
show_recommendations(200)  # Replace with any index from training set
```

### Recommend Based on a New Image
```python
new_img = x_test[42]  # Replace with your own 28x28 grayscale image
recommend_for_new_image(new_img)
```

### Recommend by Category
```python
show_one_per_class()  # Display one sample per class
user_label = int(input("Enter a label (0–9): "))
query_index = get_random_index_for_label(user_label)
show_recommendations(query_index)
```

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

---

## Acknowledgments

- **Fashion MNIST** dataset by Zalando Research  
  https://github.com/zalandoresearch/fashion-mnist

- **TensorFlow/Keras** for deep learning framework  
  https://www.tensorflow.org/

- **Scikit-learn** for similarity metrics  
  https://scikit-learn.org/



