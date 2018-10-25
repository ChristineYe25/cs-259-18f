#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>
using namespace std;


const int memory_size=10;
const int kBurstSize=180;
const int kTileSize=10;

void Load (const bool enable,unsigned long* data_dram, unsigned long* data_local){
#pragma HLS inline off
    if(enable){
load:
    for(int i=0;i<kBurstSize;++i){
#pragma HLS pipeline
        data_local[i]=data_dram[i];
    }
    }
    
}
void Compute(const bool enable,unsigned* data_local, unsigned long test_image){
#pragma HLS inline off
    if(enable){
    for(int i=0;i<kBurstSize/kTileSize;++i) {
#pragma HLS pipeline
        for(int j=0;j<kTileSize;++j){
#pragma HLS unroll
            data_local[i*kTileSize+j]=data_local[i*kTileSize+j]^test_image
            unsigned long dis=0;
            for(int z=0;z<49;++z){
                dis+=(data_local[i*kTileSize+j] & (1L<<z))>>z;
            }
            data_local[i*kTileSize]+j]=dis;
        }
    }
}
}

template<int n>
void Reduce(unsigned long *array){
#pragma HLS inline off
reduce:
    for(int i=0;i<n;++i){
#pragma HLS pipeline
        array[i]+=array[i+n];
    }
}

void Update (unsigned char* knn_mat,unsigned long* data_local,int x){
#pragma HLS inline off
unsigned long max_id=0;
update:
    for (int m=0;m<memory_size;m++){
#pragma HLS pipeline
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
}
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
    const int kMinTripCount=1;
    const int kMaxTripCount=kMinTripCount+1800/kBurstSize;
    unsigned long data_local_0[kBurstSize];
    unsigned long data_local_1[kBurstSize];
#pragma HLS array_partition variable = data_local_0 cyclic factor = kTileSize
#pragma HLS array_partition variable = data_local_1 cyclic factor = kTileSize
init:
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 3; ++y) {
            // Note that the max distance is 49
            knn_mat[(y + (x * 3))] = (unsigned char)50;
        }
    }

 //computation
digit:
   for(int i=0;i<10;++i,train_images+=1800){
       for(int j=0;j<1800+kBurstSize;j+=kBurstSize,train_images+=kBurstSize){
#pragma HLS loop_tripcount min = kMinTripCount max = kMaxTripCount
           if((j/kBurstSize)%2){
               Load(j<1800,train_images,data_local_0);
               Compute(j>0,data_local_1,test_image);
           }
           else {
               Load(j<1800,train_images,data_local_1);
               Compute(j>0,data_local_0,test_image);
           }
       }
    }
 
}
//
update:
    for (int x3 = 0; x3 < 10; ++x3) {
        for (int y3 = 0; y3 < 1800; ++y3) {
            unsigned long max_id = 0;
            for (int i1 = 0; i1 < 3; ++i1) {
                if (knn_mat[max_id + (x3 * 3)] < knn_mat[(i1 + (x3 * 3))]) {
                    max_id = i1;
                }
            }
            if (temp[y3 + (x3 * 1800)] < knn_mat[max_id + (x3 * 3)]) {
                knn_mat[max_id + (x3 * 3)] = temp[y3 + (x3 * 1800)];
            }
        }
    }

} // extern "C"
