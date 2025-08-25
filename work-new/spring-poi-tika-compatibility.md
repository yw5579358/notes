
# Spring Boot + Tika + POI 文档解析与导出兼容性问题记录

## 问题背景

在项目中同时使用了以下功能模块：

- **文件上传与解析**（依赖 Apache Tika）
- **Markdown 转 Word 导出**（依赖 poi-tl）
- **Spring AI 文档处理**（spring-ai-tika）

在引入这些功能模块后，开始出现了各种奇怪且繁杂的问题，包括但不限于：

- `NoClassDefFoundError`：类在运行时无法找到
- `NoSuchFieldError`：POI 中某些字段在新旧版本间不兼容
- **OCR 异常启动**：Tika 默认启用了 OCR 功能，造成性能问题甚至假死
- **堆栈溢出（StackOverflowError）**：上传较大或特殊 `.docx` 文件直接触发错误
- **部分 `.doc` 能用，`.docx` 失败，或反之**：不同格式支持度不一致

整个过程中最初以为是 Spring AI Tika 与 poi-tl 的冲突，实际上 Tika 自身也依赖了多个 POI 模块，而 poi-tl 默认使用了老版本 POI（4.x），而 Tika 使用的是新版本（5.x），造成依赖地狱。

##  排查过程概述

整个排查过程历时较长，期间尝试了如下方式：

- 替换 POI 版本（4.1.2、5.0.0、5.2.3、5.4.0 均试过）
- 修改 poi-tl 版本（从 1.10 到最新 1.12.2）
- 去除 OCR 功能（通过设置 TikaContext）避免触发无用的识别流程
- 查看 Spring AI Tika 实现源码，确认其使用 `AutoDetectParser` 以及相关 `TikaConfig` 行为
- 分析 `spring-webmvc` 中的异常栈排查控制器异常传播
- 使用 `mvn dependency:tree` 追踪依赖冲突，发现 tika-core、poi-ooxml、poi-tl 中多版本并存

即便替换 POI 版本后仍有问题，期间遇到的问题原因主要包括：

- POI 本身模块化程度较低，多个模块之间强依赖，某一个类变动就影响全局
- 旧版 poi-tl 与新版 POI 类名、方法字段签名完全不兼容
- Tika 使用反射 + SPI 加载类，调试起来困难
- 错误经常是运行期才暴露，编译期无提示，造成定位难
- 除去tika 与 poi-tl直接重叠依赖需要保持一致(最开始方式)，还需找出各自自身所引用的其它poi依赖并保持一致
- 网上暂无相关问题及解决方案，需逐个尝试查找问题

##  最终解决方案

### 1. 统一使用 **POI 5.4.0**

```xml
<dependency>
    <groupId>org.apache.poi</groupId>
    <artifactId>poi</artifactId>
    <version>5.4.0</version>
</dependency>
<dependency>
    <groupId>org.apache.poi</groupId>
    <artifactId>poi-ooxml</artifactId>
    <version>5.4.0</version>
</dependency>
<dependency>
    <groupId>org.apache.poi</groupId>
    <artifactId>poi-scratchpad</artifactId>
    <version>5.4.0</version>
</dependency>
```

### 2. 升级 `poi-tl` 至 **1.12.2**

```xml
<dependency>
    <groupId>com.deepoove</groupId>
    <artifactId>poi-tl</artifactId>
    <version>1.12.2</version>
</dependency>
```

### 3. 引入兼容的 commons 依赖（用于 Tika）

```xml
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-compress</artifactId>
    <version>1.27.1</version>
</dependency>
<dependency>
    <groupId>commons-io</groupId>
    <artifactId>commons-io</artifactId>
    <version>2.16.1</version>
</dependency>
```

## 当前使用的相关依赖列表

| 依赖名称                 | 版本       | 用途说明                         |
|--------------------------|------------|----------------------------------|
| `poi`                   | 5.4.0      | Office 文档通用解析              |
| `poi-ooxml`             | 5.4.0      | `.docx`, `.xlsx` 支持            |
| `poi-scratchpad`        | 5.4.0      | `.doc`, `.xls` 支持              |
| `poi-tl`                | 1.12.2     | Markdown / 模板导出 Word        |
| `poi-tl-plugin-markdown` | 1.0.3     | Markdown 转 Word 支持插件       |
| `spring-ai-tika-document-reader` | 1.0.0 | 文档读取（用于 AI 知识库）       |
| `commons-compress`      | 1.27.1     | Tika 解压依赖                    |
| `commons-io`            | 2.16.1     | 文件流读取                       |

##  注意事项

- 避免 Tika 与 poi-tl 使用不同 POI 版本。
- 不建议让 Maven 自动引入 transitive 版本，使用 `dependencyManagement` 控制版本。
- OCR 功能若无使用需求，建议手动禁用。
- Markdown 转 Word 建议优先转为 HTML，再渲染到 Word 模板，稳定性更好。
