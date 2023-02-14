def main():
    raise NotImplementedError("""BestOfN数据集预处理工作

* 原始 json 文件为 {"prompt": "text","samples": ["text1", "text2", "text3", "text4"], "label": 0-3}
* 生成的 数据集文件
    * prompt{.bin, .idx}
    * sample.0{.bin, .idx}
    * sample.1{.bin, .idx}
    * sample.2{.bin, .idx}
    * sample.3{.bin, .idx}
    * label{.bin, .idx}

""")


if __name__ == '__main__':
    main()
