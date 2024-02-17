from Embeddings import classEmbdding

embedding = classEmbdding(chroma_url="./Chroma", data_url="./", subfolder="Data")
embedding.create_vector_db_Indvdl_files()