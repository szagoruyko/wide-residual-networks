WRN-50-2
==========

Best performing ImageNet model from Wide Residual Networks BMVC 2016 paper https://arxiv.org/abs/1605.07146<br>
The model is slower than ResNet-101 and faster than ResNet-152, with better accuracy:

| Model | top-1 err, % | top-5 err, % | #params | time/batch 16 |
|---|---|---|---|---|
| ResNet-50 | 24.01 | 7.02 | 25.6M | 49 |
| ResNet-101 | 22.44 | 6.21 | 44.5M | 82 |
| ResNet-152 | 22.16 | 6.16 | 60.2M | 115 |
| __WRN-50-2-bottleneck__ | 21.9 | 6.03 | 68.9M | 93 |
| pre-ResNet-200 | 21.66 | 5.79 | 64.7M | 154 |

Download (263MB): https://yadi.sk/d/-8AWymOPyVZns

PyTorch and Tensorflow pretrained weights and model definitions:<br>
<https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb>

Convergence plot:

![bokeh_plot 4](https://cloud.githubusercontent.com/assets/4953728/20243021/98a2a66a-a945-11e6-807b-a037f667c052.png)

If you find this model useful please cite this paper:

```bib
@INPROCEEDINGS{Zagoruyko2016WRN,
    author = {Sergey Zagoruyko and Nikos Komodakis},
    title = {Wide Residual Networks},
    booktitle = {BMVC},
    year = {2016},
}
```


# Model printout

```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> output]
  (1): cudnn.SpatialConvolution(3 -> 64, 7x7, 2,2, 3,3) without bias
  (2): nn.SpatialBatchNormalization (4D) (64)
  (3): cudnn.ReLU
  (4): nn.SpatialMaxPooling(3x3, 2,2, 1,1)
  (5): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(64 -> 128, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (128)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (128)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(128 -> 256, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (256)
          |    }
           `-> (2): nn.Sequential {
                 [input -> (1) -> (2) -> output]
                 (1): cudnn.SpatialConvolution(64 -> 256, 1x1) without bias
                 (2): nn.SpatialBatchNormalization (4D) (256)
               }
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (2): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(256 -> 128, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (128)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (128)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(128 -> 256, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (256)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (3): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(256 -> 128, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (128)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (128)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(128 -> 256, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (256)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
  }
  (6): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> output]
    (1): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(256 -> 256, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (256)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(256 -> 256, 3x3, 2,2, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (256)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(256 -> 512, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (512)
          |    }
           `-> (2): nn.Sequential {
                 [input -> (1) -> (2) -> output]
                 (1): cudnn.SpatialConvolution(256 -> 512, 1x1, 2,2) without bias
                 (2): nn.SpatialBatchNormalization (4D) (512)
               }
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (2): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(512 -> 256, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (256)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (256)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(256 -> 512, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (512)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (3): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(512 -> 256, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (256)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (256)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(256 -> 512, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (512)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (4): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(512 -> 256, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (256)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (256)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(256 -> 512, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (512)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
  }
  (7): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    (1): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(512 -> 512, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (512)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(512 -> 512, 3x3, 2,2, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (512)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(512 -> 1024, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (1024)
          |    }
           `-> (2): nn.Sequential {
                 [input -> (1) -> (2) -> output]
                 (1): cudnn.SpatialConvolution(512 -> 1024, 1x1, 2,2) without bias
                 (2): nn.SpatialBatchNormalization (4D) (1024)
               }
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (2): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(1024 -> 512, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (512)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (512)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(512 -> 1024, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (1024)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (3): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(1024 -> 512, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (512)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (512)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(512 -> 1024, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (1024)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (4): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(1024 -> 512, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (512)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (512)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(512 -> 1024, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (1024)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (5): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(1024 -> 512, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (512)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (512)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(512 -> 1024, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (1024)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (6): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(1024 -> 512, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (512)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (512)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(512 -> 1024, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (1024)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
  }
  (8): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(1024 -> 1024, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (1024)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(1024 -> 1024, 3x3, 2,2, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (1024)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(1024 -> 2048, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (2048)
          |    }
           `-> (2): nn.Sequential {
                 [input -> (1) -> (2) -> output]
                 (1): cudnn.SpatialConvolution(1024 -> 2048, 1x1, 2,2) without bias
                 (2): nn.SpatialBatchNormalization (4D) (2048)
               }
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (2): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(2048 -> 1024, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (1024)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(1024 -> 1024, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (1024)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(1024 -> 2048, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (2048)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
    (3): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): cudnn.SpatialConvolution(2048 -> 1024, 1x1) without bias
          |      (2): nn.SpatialBatchNormalization (4D) (1024)
          |      (3): cudnn.ReLU
          |      (4): cudnn.SpatialConvolution(1024 -> 1024, 3x3, 1,1, 1,1) without bias
          |      (5): nn.SpatialBatchNormalization (4D) (1024)
          |      (6): cudnn.ReLU
          |      (7): cudnn.SpatialConvolution(1024 -> 2048, 1x1) without bias
          |      (8): nn.SpatialBatchNormalization (4D) (2048)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.CAddTable
      (3): cudnn.ReLU
    }
  }
  (9): cudnn.SpatialAveragePooling(7x7, 1,1)
  (10): nn.View(2048)
  (11): nn.Linear(2048 -> 1000)
}
```
