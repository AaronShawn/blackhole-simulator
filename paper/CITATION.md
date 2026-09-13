# 引用配套论文 / Citing the companion paper

首选引用形式是**软件 + 论文**一并引用：软件条目见仓库根目录
[`CITATION.cff`](../CITATION.cff)（可按 GitHub 的 *Cite this repository* 按钮直接导出）。

## BibTeX

```bibtex
@techreport{blackholeSimulator2026paper,
  title       = {施瓦西黑洞实时成像模拟器：物理建模、数值方法与 GPU 实现},
  author      = {徐上},
  institution = {无锡工艺职业技术学院},
  type        = {技术报告},
  year        = {2026},
  month       = {9},
  version     = {1.0.1},
  note        = {25 图；19 表。源代码与全部生成脚本：
                 https://github.com/AaronShawn/blackhole-simulator},
  url         = {https://github.com/AaronShawn/blackhole-simulator/blob/main/paper/paper.pdf}
}
```

英文题录（若投稿要求英文条目）：

```bibtex
@techreport{blackholeSimulator2026paperEn,
  title       = {A Real-Time {Schwarzschild} Black-Hole Imaging Simulator:
                 Physical Modelling, Numerical Methods and {GPU} Implementation},
  author      = {徐上},
  institution = {无锡工艺职业技术学院},
  year        = {2026},
  month       = {9},
  version     = {1.0.1},
  url         = {https://github.com/AaronShawn/blackhole-simulator/blob/main/paper/paper.pdf}
}
```

## 论文覆盖的内容

| 章 | 主题 | 该章产物 |
| --- | --- | --- |
| §1 引言 | 研究背景；与 Luminet (1979)、EHT 观测、已有实时渲染工具的定位；本文贡献 | — |
| §2 物理模型 | 施瓦西度规、零测地线第一积分、临界冲击参数 $b_c=3\sqrt3M$ 与阴影张角、偏折角弱场极限、Page–Thorne 与 Shakura–Sunyaev 剖面对照、频移与强度变换 | 图 1、2、7、9、10；表 7–10 |
| §3 数值方法 | 像素→轨道平面归约、RK4 格式与误差阶、几何自适应步长、终止判据与光线分类、体积辐射转移、守恒残差 | 表 15 |
| §4 实现 | 四段 GPU 管线、分辨率预算与渐进累积、后处理系数、程序化星场、开普勒剪切湍流、桌面打包、UI 中心不变量 | 图 18、23；表 13、14 |
| §5 验证 | 六组实验 V1–V6：临界冲击参数反解、首次积分守恒、阴影半径像素级反解、观测者标架、有限逃逸半径、累积确定性 | 图 5、14、17、24；表 1–6、16、17 |
| §6 结果 | 代价模型 $T\approx kP+c$、像阶次统计、临界曲线对数窄带、观测者距离扫描、应用实拍画廊、参数扫描、视口自适应 | 图 16、19、20、22、25；表 11、18 |
| §7 讨论 | 施瓦西→Kerr 的适用边界、辐射转移与星场近似、计时量化、代价模型边界、「视觉说服力」掩盖的缺陷、相对定位 | — |
| §8 结论 | 结论、适用范围边界、交付物清单 | — |
| 附录 A | 符号索引、完整复现命令、仓库结构、GLSL 代码清单、容差扫描完整数据、补充插图 | 图 3、4、6、8、11、12、13、15、21；表 12；`code/*.lst` |

## 复现方式

论文中没有任何手抄数字：所有图表由 `tools/` 中的脚本从源码或实测数据生成，
`paper.pdf` 本身由 `tectonic`/`latexmk` 编译。

```powershell
cd paper
python tools/measure_probe.py       # 亚像素阴影探针（写入 tools/r0_scan.json）
python tools/winding_divergence.py  # 近临界缠绕拟合
python tools/derive_perf.py         # 性能代价模型标定
python tools/make_tables.py         # -> tables/*.tex
python tools/make_figures.py; python tools/make_figures2.py   # -> figures/
python tools/make_render_figures.py # 用真实渲染器截图生成应用类插图
python tools/extract_snippets.py    # -> code/*.lst

# 编译（tectonic 便携版，无需安装 TeX 发行版）
tectonic -X compile paper.tex --outdir . --keep-logs
```
