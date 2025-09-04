import numpy as np

print("NumPy Basics Demonstration")

# Creating an array of zeros
zeros_array = np.zeros((2, 3))

# Creating an array of ones
ones_array = np.ones((2, 3))

# Creating a 1D array
array_1d = np.array([1, 2, 3, 4, 5])

# Creating a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Creating a 3D array
array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Array attributes
for i, array in enumerate([zeros_array,ones_array,array_1d, array_2d, array_3d], start=1):
    print(f"\nArray {array}:")
    print(f"\t    Array: {array.shape}")
    print(f"\t   Shape: {array.shape}")
    print(f"\t    Data Type: {array.dtype}")
    print(f"\t    Dimensions: {array.ndim}")
    print(f"    Size: {array.size}")
    
    #Demo Array math
    print(f"    Adding Arrays: {array + array}")  # Adding two arrays

    print(f"    Multiplying Arrays By 5: {array * 5}")  # Multiplying arrays
    print(f"    Square Root of Array: {np.sqrt(array)}")  # Square root of each element
    print(f"    Exponential of Array: {np.exp(array)}")  # Exponential of each element
    print(f"    Max of Array: {np.max(array)}")  # Maximum value in the array
    print(f"    Min of Array: {np.min(array)}")  # Minimum value in the array
    print(f"    Mean of Array: {np.mean(array)}")  # Mean of the array
    print(f"    Standard Deviation of Array: {np.std(array)}")  # Standard deviation of the array
    print(f"    Sum of Array: {np.sum(array)}")  # Sum of all elements in the array