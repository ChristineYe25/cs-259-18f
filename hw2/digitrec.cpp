#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>
using namespace std;


const int memory_size=10;
const int kBurstSize=180;
const int kTileSize=18;

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
void Compute(const bool enable,unsigned long* data_local, unsigned long test_image,int x,unsigned char* min){
#pragma HLS inline off
    if(enable){
    for(int i=0;i<kBurstSize/kTileSize;++i) {
#pragma HLS pipeline
        for(int j=0;j<kTileSize;++j){
#pragma HLS unroll
            data_local[i*kTileSize+j]=data_local[i*kTileSize+j]^test_image;
            unsigned long dis=0;
           for(int z=0;z<49;++z){
                dis+=(data_local[i*kTileSize+j] & (1L<<z))>>z;
            }
             data_local[i*kTileSize+j]=dis;
            unsigned int max_id=0;
            for(int z=0;z<3;z++){
                if(min[max_id]<min[z]){
                    max_id=z;
                }
                if(dis[j]<min[max_id]){
                    min[max_id]=dis[j];
                    
                }
            }
        }
       
        
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
    unsigned char min[3];
#pragma HLS array_partition variable = data_local_0 cyclic factor = kTileSize
#pragma HLS array_partition variable = data_local_1 cyclic factor = kTileSize
 //computation
digit:
   for(int i=0;i<10;++i){
       for(int mi=0;mi<3;++mi){
           min[mi]=(unsigned char)50;
       }
       for(int j=0;j<1800+kBurstSize;j+=kBurstSize,train_images+=kBurstSize){
#pragma HLS loop_tripcount min = kMinTripCount max = kMaxTripCount
           if((j/kBurstSize)%2){
               Load(j<1800,train_images,data_local_0);
               Compute(j>0,data_local_1,test_image,i,min);
           }
           else {
               Load(j<1800,train_images,data_local_1);
               Compute(j>0,data_local_0,test_image,i,min);
           }
       }
       for(int z=0;z<3;z++){
           knn_mat[i*3+z]=min[z];
       }
    }
 

//

}
} // extern "C"
