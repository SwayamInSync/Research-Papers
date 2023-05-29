Here are the notations used in the FlashAttention algorithm (Refer to the original paper):

- **Q:** The query matrix. It is a matrix of size N×d, where N is the number of tokens in the input sequence and d is the dimension of each token.
- **K:** The key matrix. It is a matrix of size N×d, where N is the number of tokens in the input sequence and d is the dimension of each token.
- **V:** The value matrix. It is a matrix of size N×d, where N is the number of tokens in the input sequence and d is the dimension of each token.
- **B_r:** The blocksize for the rows of the input sequence. It is a positive integer that is less than or equal to N.
- **B_c:** The blocksize for the columns of the input sequence. It is a positive integer that is less than or equal to N.
- **M:** The size of the on-chip SRAM. It is a positive integer that is less than or equal to Br×Bc.
- **S_{ij}:** The matrix product QiKj. It is a matrix of size Br×Bc.
- **m_i:** The maximum of each row of Sij. It is a vector of size Br.
- **p_ij:** The exponential of each entry in Sij−mi. It is a matrix of size Br×Bc.
- **l_i:** The row sums of pij. It is a vector of size Br.
- **o_i:** The output matrix Oi=diag(li)⋅PiVj. It is a matrix of size Br×Bc.