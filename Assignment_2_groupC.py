import numpy as np

np.set_printoptions(suppress=False, floatmode='fixed', precision=1)

def gram_schmidt(vectors):
    if not vectors:
        raise ValueError("The list of vectors must not be empty.")
    
    if not all(isinstance(vec, np.ndarray) for vec in vectors):
        raise ValueError("All elements of the list must be NumPy arrays.")
    
    orthogonal_basis = []
    
    for vec in vectors:
        if vec.size == 0:
            raise ValueError("The vectors must not be empty.")
        
        w = vec.copy()
        for orth_vec in orthogonal_basis:
            if orth_vec.size == 0:
                raise ValueError("The vectors in the list must not be empty.")
            
            # Checking if orth_vec is a zero vector before performing division
            if np.allclose(orth_vec, 0):
                continue
            
            # Projecting w onto orth_vec and subtract the projection from w
            w -= np.dot(w, orth_vec) / np.dot(orth_vec, orth_vec) * orth_vec
        
        # Adding the orthogonal vector to the basis if it's not the zero vector
        if not np.allclose(w, 0):
            orthogonal_basis.append(w)
        else:
            print("Zero vector encountered and skipped.")
    
    return orthogonal_basis

dimension = int(input("Enter the dimension of the vectors: "))
num_vectors = int(input("Enter the number of vectors: "))

vectors = []
for i in range(num_vectors):
    while True:
        try:
            vector_input = input(f"Enter vector {i + 1} (space-separated): ").strip()
            if not vector_input:
                print("Error: Vector input cannot be empty. Please input again.")
                continue
            
            vector = np.array(list(map(float, vector_input.split())))
            if len(vector) > dimension:
                print(f"Vector {i + 1} exceeds the dimension limit of {dimension}, taking only the first {dimension} elements.")
                vector = vector[:dimension]
            elif len(vector) < dimension:
                print(f"Error: Vector {i + 1} has fewer elements than the dimension {dimension}. Please input again.")
                continue
            vectors.append(vector)
            print(f"[{', '.join(map(str, vector))}]")
            break
        except ValueError:
            print("Invalid input. Please enter a vector with numbers only, separated by spaces.")

orthogonal_basis = gram_schmidt(vectors)

print("Orthogonal basis:")
for vector in orthogonal_basis:
    print(vector)
