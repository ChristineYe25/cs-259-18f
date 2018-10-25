#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>
using namespace std;


const int memory_size=18;

void Load (unsigned long* data_dram, unsigned long* data_local,int index){
#pragma HLS inline off
load:
    for(int i=0;i<memory_size;i++){
#pragma HLS unroll
        data_local[i]=data_dram[index+i];
    }
    
}
void Diff(unsigned long* data_local,unsigned long test_image){
#pragma HLS inline off
diff:
    for(int i=0;i<memory_size;i++){
#pragma HLS unroll
        data_local[i]=data_local[i]^test_image;
    }
}
template<int n>
void Reduce(unsigned long *array){
#pragma HLS inline off
reduce:
    for(int i=0;i<n;++i){
#pragma HLS unroll
        array[i]+=array[i+n];
    }
}
void Dis(unsigned long* data_local){
#pragma HLS inline off
dis:
    for(int m=0;m<memory_size;m++){
#pragma HLS unroll
        unsigned long dis_local[8];
        for(int i=0;i<7;i++){
            unsigned int temp=0;
            for(int j=0;j<7;j++){
                temp+=(data_local[m]&(1L<<(i*7+j)))>>(i*7+j);
            }
            dis_local[i]=temp;
        }
        dis_local[7]=0;
        Reduce<4>(dis_local);
        Reduce<2>(dis_local);
        Reduce<1>(dis_local);
       data_local[m]=dis_local[0];
    }
 
}
void Update (unsigned char* knn_mat,unsigned long* data_local,int x){
#pragma HLS inline off
unsigned long max_id=0;
update:
    for (int m=0;m<memory_size;m++){
#pragma HLS unroll
        for(int i=0;i<3;i++){
            if(knn_mat[max_id+(x*3)]<knn_mat[(i+(x*3))]){
             max_id=i;
           }
         }
        if(data_local[m]<knn_mat[max_id+(x*3)]){
            knn_mat[max_id + (x*3)]=data_local[m];
         }
    }
}
void Loop(unsigned long* train_images,unsigned long* data_local,int i,unsigned char* knn_mat,unsigned long test_image) {
loop_ins:
for(int y=0;y<1800/memory_size;y++){
#pragma HLS pipeline
    Load(train_images,data_local,i*1800+y*memory_size);
    Diff(data_local,test_image);
    Dis(data_local);
    Update(knn_mat,data_local,i);
              
          }
)
extern "C" {
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
  unsigned long data_local[memory_size];
#pragma HLS array_partition variable = data_local cyclic factor = memory_size
    
init:
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 3; ++y) {
            // Note that the max distance is 49
            knn_mat[(y + (x * 3))] = (unsigned char)50;
        }
    }

 //the 10 digit loop
digit:
   for(int i=0;i<10;i++){
       Loop(train_images,data_local,i,knn_mat,test_image);
    }
 
}

} // extern "C"
