#include "data_utils.cuh"
#include "fused_kernels.cuh"
#include "wmma_kernels.cuh"
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
template<class T> vector<T> readv(FILE*f,int n){vector<T> v(n);if(fread(v.data(),sizeof(T),n,f)!=(size_t)n)exit(3);return v;}
template<class T> T* upload(const vector<T>&v){T*p;CK(cudaMalloc(&p,v.size()*sizeof(T)));CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char**argv){
 if(argc!=3)return 1;FILE*f=fopen(argv[1],"rb");if(!f)return 1;
 auto h=readv<int>(f,4);int n=h[0],nf=h[1],k=h[2],nnz=h[3];if(k!=K_DIM)return 4;
 auto off=readv<int>(f,n+1),idx=readv<int>(f,nnz);auto val=readv<float>(f,nnz),feat=readv<float>(f,nf*k);fclose(f);
 auto od=upload(off),id=upload(idx);auto vd=upload(val),fd=upload(feat);float *lhs,*rhs;half*fh;
 CK(cudaMalloc(&lhs,n*k*k*sizeof(float)));CK(cudaMalloc(&rhs,n*k*sizeof(float)));CK(cudaMalloc(&fh,nf*k*sizeof(half)));
 convert_fp32_to_fp16<<<(nf*k+255)/256,256>>>(fd,fh,nf*k);
 vector<int>tp,tc,sp,sc,nzl,nzp;vector<float>sv,den;
 int yb=k>32?128:YB;
 build_BALS_format(n,nf,XB,yb,off,idx,val,tp,tc,sp,sc,sv,den,nzl,nzp);fill(den.begin(),den.end(),0.f);
 vector<int>jt,jc,jn;
 for(int tx=0;tx<(n+XB-1)/XB;++tx){int c=(nzp[tx+1]-nzp[tx]+TILES_PER_CHUNK-1)/TILES_PER_CHUNK;for(int j=0;j<c;++j){jt.push_back(tx);jc.push_back(j);jn.push_back(c);}}
 auto tpd=upload(tp),tcd=upload(tc),spd=upload(sp),scd=upload(sc),nzld=upload(nzl),nzpd=upload(nzp),jtd=upload(jt),jcd=upload(jc),jnd=upload(jn);auto svd=upload(sv),dd=upload(den);
 constexpr int rr=(K_DIM==48||K_DIM==96)?3:2, tr=K_DIM/rr, dz=1024/(tr*tr), split=K_DIM==16?1:K_DIM==32?2:K_DIM==48?4:8, rpt=XB/(dz*split);
 int smem=yb*k*sizeof(float);
 CK(cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_block<rpt,rr,rr>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 vector<float>out(n*k*k);FILE*g=fopen(argv[2],"wb");if(!g)return 1;
 for(int mode=0;mode<4;++mode){
 CK(cudaMemset(lhs,0,n*k*k*sizeof(float)));CK(cudaMemset(rhs,0,n*k*sizeof(float)));
 if(mode==0)compute_LHS_RHS_BALS_block<rpt,rr,rr><<<dim3(jt.size(),split),dim3(tr,tr,dz),smem>>>(n,k,0,tpd,tcd,spd,scd,svd,fd,lhs,rhs,dd,nzld,nzpd,jtd,jcd,jnd,0,n);
 if(mode==1)compute_LHS_RHS_wmma<K_DIM/16><<<(n+WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK,WARPS_PER_BLOCK*32>>>(n,0,od,id,vd,fh,lhs,rhs,0);
 if(mode==2)compute_LHS_RHS_wmma_tf32<K_DIM/16><<<(n+WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK,WARPS_PER_BLOCK*32>>>(n,0,od,id,vd,fd,lhs,rhs,0);
 if(mode==3)compute_LHS_RHS_wmma_giant<K_DIM/16,GIANT_WARPS><<<n,GIANT_WARPS*32>>>(n,od,id,vd,fh,lhs,rhs,0,0);
 CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(out.data(),lhs,out.size()*sizeof(float),cudaMemcpyDeviceToHost));fwrite(out.data(),sizeof(float),out.size(),g);
 }
 fclose(g);printf("Wrote atomic, FP16 dense, FP32 sameflow, FP16 giant; %d entities, K=%d\n",n,k);
}
