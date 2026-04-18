# 如何贡献这个维基？

[项目 GitHub 地址](https://github.com/yfrobotics/self-driving-handbook-cn)

## 贡献方式

你可以通过以下任意方式参与：

- **提交 Pull Request**：fork → clone → 修改 → push → 发起 PR
- **提交 Issue**：对内容、结构、错别字等任何问题[创建 issue](https://github.com/yfrobotics/self-driving-handbook-cn/issues)

在动手前，建议先阅读[书写规范](standard.md)，新增条目时请同步更新 `mkdocs.yml` 的 `nav` 配置。

## 本地开发

推荐使用 Python 虚拟环境，避免与系统环境冲突：

```bash
# 1. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动本地预览（默认 http://127.0.0.1:8000）
mkdocs serve

# 4. 严格构建（与 CI 一致，会在有警告时报错）
mkdocs build --strict
```

构建产物位于 `site/` 目录，已加入 `.gitignore`。

## 新增页面

1. 在 `docs/` 下对应章节目录创建 `.md` 文件；
2. 打开 `mkdocs.yml`，在 `nav` 中加入新文件的引用；
3. 运行 `mkdocs serve` 验证导航与渲染；
4. 提交 PR 时附带简短说明，便于评审。

## 推荐工具

- **Markdown 编辑**：Visual Studio Code（配合 Markdown All in One 插件）、Obsidian、Zed
- **Git 图形化**：命令行 `git`、GitHub Desktop、SourceTree、GitKraken
- **预览**：本地 `mkdocs serve`；也可在 GitHub 上直接预览单个 Markdown 文件

## 参考

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs 参考](https://squidfunk.github.io/mkdocs-material/reference/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)
