#include <string.h>
#include <math.h>
#include <assert.h>

extern "C" {
const int memory_size=100;

void Load (unsigned long* data_dram, unsigned long* data_local){
#pragma HLS inline off
    for(int i=0;i<memory_size;i++){
#pragma HLS unroll
        data_local[i]=data_dram[i];
    }
}
void Diff(unsigned long* data_local,unsigned long test_image){
#pragma HLS inline off
    for(int i=0;i<memory_size;i++){
#pragma HLS unroll
        data_local[i]=data_local[i]^test_image;
    }
}
template<int n>
void Reduce(unsigned long *array,int start){
#pragma HLS inline off
    for(int i=start;i<n;++i){
#pragma HLS unroll
        array[i]+=array[i+n];
    }
}
void Dis(unsigned long* data_local){
#pragma HLS inline off
    unsigned long dis_local[8*memory_size];
    for(int m=0;m<memory_size;m++){
#pragma HLS unroll
        for(int i=0;i<8;i++){
#pragma HLS unroll
            if(i==7) {
                dis_local[i+m*49]=0;
                break;
            }
            for(int j=0;j<7;j++){
#pragma HLS pipeline
                dis_local[i+m*49]+=(data_local[m]&(1L<<(i*7+j)))>>(i*7+j);
            }
        }
        Reduct<4>(dis_local,m*49);
        Reduct<2>(dis_local,m*49);
        Reduce<1>(dis_local,m*49);
       data_local[m]=dis_local[m*49];
    }
}
void Update (unsigned long* knn_mat,unsigned long* data_local,int x){
    unsigned long max_id;
    for (int m=0;m<memory_size;m++){
#pragma HLS unroll
        for(int i=0;i<3;i++){
            if(knn_mat[max_id]+(x*3)<knn_mat[(i+(x*3))]){
             max_id=i;
           }
         }
        if(data_local[m]<knn_mat[max_id+(x*3)]){
            knn_mat[max_id + (x3*3)]=data_local[m];
         }
    }
}

void digitrec_kernel(
    unsigned long test_image,
    unsigned long* train_images,
    unsigned char* knn_mat) {
#pragma HLS interface m_axi port=train_images offset=slave bundle=gmem
#pragma HLS interface m_axi port=knn_mat offset=slave bundle=gmem2
#pragma HLS interface s_axilite port=test_image bundle=control
#pragma HLS interface s_axilite port=train_images bundle=control
#pragma HLS interface s_axilite port=knn_mat bundle=control
#pragma HLS interface s_axilite port=return bundle=control

  unsigned int a;
  unsigned data_local[memory_size];
#pragma HLS array_partition variable = data_local cyclic factor = memory_size
    
init:
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 3; ++y) {
            // Note that the max distance is 49
            knn_mat[(y + (x * 3))] = (unsigned char)50;
        }
    }

 //the 10 digit loop
    for(int i=0;i<10;i++){
#pragma HLS pipeline
        for(int y=0;y<1800/memory_size;y++){
#pragma HLS pipeline
            Load(train_images[i*1800+y*memory_size+z],data_local);
            Diff(data_local,test_image);
            Dis(data_local);
            Update(data_local);
            
        }
    }
}

} // extern "C"
