
#ifndef COPY_CUH
#define COPY_CUH

// chunk_size = 8, dim = 128, float16 ==> 8*128*2=2048 Bytes
// PTYPE = int2 (8 Bytes) ==> BLOCK_SIZE_CP = 256 (one chunk has 256 int2 data)
#define CHUNK_DATA_SIZE (CHUNK_SIZE * 128 * DATA_BYTES)
#define BLOCK_SIZE_CP (CHUNK_DATA_SIZE / sizeof(PTYPE))

// one vector has (128*2/8)=32 int2 data
#define VECTOR_SIZE_CP ((128 * DATA_BYTES) / sizeof(PTYPE))

#endif // COPY_CUH
