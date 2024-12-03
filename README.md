# demucs.axcl
demucs on Axera PCIE-card demo

音乐分离模型，[官方repo](https://github.com/facebookresearch/demucs)

## 下载模型
```
./download_models.sh
```

## 编译

- 支持在 x86_64、aarch64 平台下本地编译
- 支持在 x86_64 下交叉编译生成 aarch64 环境中可运行的程序

x86_64
```
./build.sh
```

aarch64
```
./build_aarch64.sh
```

交叉编译
```
./cross_compile.sh
```

## 运行
```
./install/demucs -m ../models/htdemucs_ft.axmodel -i 输入音频.wav -o 输出目录
```