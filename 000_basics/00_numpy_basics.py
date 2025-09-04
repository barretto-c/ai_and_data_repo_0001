import numpy as np

print("NumPy Basics Demonstration")

# Creating a 1D array
array_1d = np.array([1, 2, 3, 4, 5])

# Creating a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Creating a 3D array
array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Array attributes
for i, array in enumerate([array_1d, array_2d, array_3d], start=1):
    print(f"\nArray {array}:")
    print(f"    Array: {array.shape}")
    print(f"    Shape: {array.shape}")
    print(f"    Data Type: {array.dtype}")
    print(f"    Dimensions: {array.ndim}")
    print(f"    Size: {array.size}")
    
    #Demo Array math
    print(f"    Adding Arrays: {array + array}")  # Adding two arrays

    print(f"    Multiplying Arrays By 5: {array * 5}")  # Multiplying arrays
    print(f"    Square Root of Array: {np.sqrt(array)}")  # Square root of each element
    print(f"    Exponential of Array: {np.exp(array)}")  # Exponential of each element
    print(f"    Max of Array: {np.max(array)}")  # Maximum value in the array


# Creating arrays with specific values
zeros_array = np.zeros((2, 3))
print("Zeros Array:\n", zeros_array)

ones_array = np.ones((2, 3))
print("Ones Array:\n", ones_array)

