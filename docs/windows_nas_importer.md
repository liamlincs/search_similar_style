# Windows NAS 款图批量入库工具

## 1. 映射 NAS 盘符

在 Windows 命令行执行，按实际 NAS 局域网 IP 和密码修改：

```bat
net use Z: \\192.168.1.50\photo /user:code_user 你的NAS密码 /persistent:yes
```

映射后路径示例：

```text
Z:\2018\成衣
Z:\products\standard_samples
```

## 2. 启动工具

在项目目录双击：

```text
scripts\windows\run_nas_importer.bat
```

或手动运行：

```bat
.venv\Scripts\python.exe tools\windows_nas_importer.py --target "Z:\products\standard_samples"
```

浏览器打开：

```text
http://127.0.0.1:7860/
```

## 3. 使用方式

- 源目录在页面输入，例如 `Z:\2018\成衣`。
- 目标目录默认 `Z:\products\standard_samples`，可以手工修改。
- 源目录和目标目录会保存在浏览器本地，下次打开自动填入。
- 点击“扫描识别”后，按钮会变灰并显示扫描中；需要中断时点旁边的“停止扫描”。
- 页面顶部会显示“已处理 x/y”和进度条。
- 工具会 OCR 识别款号。
- OCR 失败或格式异常的行会标红，可以人工改款号。
- 页面默认不加载图片预览；需要查看时点击源文件名弹出预览，避免一次性读取大量 NAS 图片。
- 人工修改款号后，“导入后文件名”会自动跟着改。
- 批量标签区提供年份、分类、细类三个一级标签输入和快捷按钮。
- 年份只从批量标签区填写，不再按每行款号自动提取。
- 三个标签栏里手工输入的新标签会保存在浏览器本地，后续作为快捷标签显示。
- 点击“确认复制入库”后，图片会复制到目标目录。

导入工具会在目标目录写入 `_nas_import_manifest.jsonl` 标签清单。主服务器凌晨定时维护会先同步产品库、读取这份清单写入标签，再刷新主特征库；新图片会进入图搜。
