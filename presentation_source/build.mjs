import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const root=path.resolve('..');
const build=path.resolve('.');
const out=path.join(root,'开题报告');
const content=JSON.parse(await fs.readFile(path.join(build,'content.json'),'utf8'));
const SKILL_DIR='C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const PY='C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const C={white:'#FFFFFF',navy:'#12365A',blue:'#1674CE',cyan:'#008EBD',ink:'#233D54',muted:'#5C7084',pale:'#EEF6FD',line:'#CCDDEB'};
const font='Microsoft YaHei';
const p=Presentation.create({slideSize:{width:1280,height:720}});
function txt(s,t,x,y,w,h,size=28,color=C.ink,bold=false,align='left'){
  const o=s.shapes.add({geometry:'textbox',name:t.slice(0,40),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  o.text=t; o.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',insets:0,wrap:'none',autoFit:'none'};
  return o;
}
function rect(s,x,y,w,h,fill=C.pale,stroke='none',sw=0){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:sw}});}
function rule(s,x,y,w,color=C.line,th=1){return rect(s,x,y,w,th,color);}
function line(s,x,y,w,h,color=C.blue){return s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width:2}});}
function box(s,label,x,y,w,h,{fill=C.pale,color=C.navy,size=26,bold=true}={}){
  const o=rect(s,x,y,w,h,fill,C.line,1);
  o.text=label;o.text.style={typeface:font,fontSize:size,bold,color,alignment:'center',verticalAlignment:'middle',insets:12,autoFit:'none'};return o;
}
function connect(s,a,b,from='right',to='left',color=C.blue){
  return s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:color,width:2},tail:{type:'triangle',width:'sm',length:'sm'}});
}
function base(i,title,subtitle=''){
  const s=p.slides.add();s.background.fill=C.white;
  txt(s,title,64,40,1152,68,42,C.navy,true);
  if(subtitle)txt(s,subtitle,66,114,1150,38,23,C.muted);
  rule(s,64,164,1152);
  rule(s,64,665,1152);
  txt(s,'电子系统设计 · 开题报告',64,678,900,24,16,C.muted);
  txt(s,String(i).padStart(2,'0'),1158,676,58,28,19,C.blue,true,'right');
  s.speakerNotes.text=content.slides[i-1].speech+'\n\n资料依据：\n'+content.sources.join('\n');
  return s;
}
function table(s,values,widths,x=64,y=190,h=410,size=25){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:widths.reduce((a,b)=>a+b,0),height:h,columnWidths:widths,values});
  t.borders.assign({fill:C.line,width:1,style:'solid'});
  for(let r=0;r<values.length;r++){
    t.rows[r].height=h/values.length;
    for(let c=0;c<values[r].length;c++){
      const cell=t.getCell(r,c);cell.fill=r===0?C.navy:(r%2===1?C.white:C.pale);
      cell.text.style={typeface:font,fontSize:size,bold:r===0||c===0,color:r===0?C.white:(c===0?C.navy:C.ink),verticalAlignment:'middle',alignment:'left',insets:{left:16,right:12,top:8,bottom:8},autoFit:'none'};
    }
  }
  return t;
}

// 01 Cover: editable title on a predominantly white field.
{
 const s=p.slides.add();s.background.fill=C.white;
 txt(s,'电子系统设计  /  开题报告',70,65,1120,40,25,C.blue,true);
 txt(s,'基于昇腾平台的',70,197,1110,76,57,C.navy,true);
 txt(s,'多模态交互自主跟随与',70,278,1110,76,57,C.navy,true);
 txt(s,'物品抓取机器人设计',70,359,1110,76,57,C.navy,true);
 rule(s,72,475,1136,C.blue,3);
 txt(s,'Huawei Atlas 200I DK A2',72,512,1110,52,32,C.blue,true);
 txt(s,'ROS 2 移动操作机器人',72,574,1110,42,26,C.muted);
 txt(s,'室内辅助场景',72,661,1110,28,19,C.muted);
 s.speakerNotes.text=content.slides[0].speech+'\n\n'+content.sources[2];
}
// 02 Context, three concrete task rows.
{
 const s=base(2,'课题背景与应用价值','面向室内轻质物品辅助任务，连接人的指令与机器人的动作');
 const rows=[['01','自主跟随','“跟着我”','保持距离移动，遇障碍减速或绕行'],['02','激光指引','“去这里”','识别地面光斑，生成可通行目标点'],['03','语音抓取','“抓起红色杯子”','识别目标物品，执行机械臂抓取']];
 rows.forEach((a,i)=>{let y=208+i*125;txt(s,a[0],66,y,65,60,41,C.blue,true);txt(s,a[1],156,y,240,51,31,C.navy,true);txt(s,a[2],435,y,380,51,29,C.ink,true);txt(s,a[3],435,y+54,750,40,25,C.muted);if(i<2)rule(s,156,y+105,1058);});
 txt(s,'应用价值：降低交互门槛，验证国产边缘推理平台上的感知与控制协同',66,604,1150,40,25,C.blue,true);
}
// 03 Native acceptance table.
{
 const s=base(3,'设计任务与验收要求','下列数值为拟定目标，均需在规定条件下实测');
 table(s,[['功能','目标指标','验收条件与方法'],['自主导航','到达率、避障通过率 ≥90%','静态障碍室内，各测试 10 次'],['人体跟随','距离误差 ≤±0.3 m','设定 1.5 m，直行与转弯 2 min'],['激光指引','定位误差 ≤0.2 m','3 m 内地面光斑，测试 10 点'],['语音控制','指令识别率 ≥90%','安静室内，1 m，20 条预定义指令'],['物品抓取','成功率 ≥70%','轻质小物体、固定作业平面，10 次']], [210,365,577],64,188,408,25);
 txt(s,'范围约束：室内低速运行，抓取纸杯、海绵块等轻物，急停优先',68,616,1140,32,23,C.muted);
}
// 04 Native system block diagram.
{
 const s=base(4,'系统总体架构','上位机承担感知与规划，下位机执行实时控制');
 txt(s,'感知输入',68,186,240,38,26,C.navy,true);
 txt(s,'Atlas 200I DK A2',395,186,500,38,28,C.blue,true);
 txt(s,'执行与反馈',997,186,240,38,26,C.navy,true);
 const camera=box(s,'USB 摄像头',64,258,235,66);
 const lidar=box(s,'二维激光雷达',64,358,235,66);
 const mic=box(s,'USB 麦克风',64,458,235,66);
 rect(s,360,239,555,309,C.white,C.line,2);
 const npu=box(s,'NPU\n视觉推理 / 抓取策略适配',392,264,491,88,{fill:C.pale,size:25});
 const cpu=box(s,'CPU / ROS 2\n语音、SLAM、Nav2、任务管理',392,387,491,110,{fill:C.pale,size:25});
 const arm=box(s,'SO-101 机械臂\n独立串口总线',975,258,241,100,{size:25});
 const esp=box(s,'ESP32-S3\n电机 PID / 编码器 / IMU',975,406,241,112,{size:23});
 for(const y of [282,382,482])s.shapes.add({geometry:'rightArrow',position:{left:307,top:y,width:45,height:19},fill:C.blue,line:{fill:'none',width:0}});
 s.shapes.add({geometry:'downArrow',position:{left:628,top:357,width:19,height:25},fill:C.blue,line:{fill:'none',width:0}});
 for(const y of [300,454])s.shapes.add({geometry:'rightArrow',position:{left:922,top:y,width:45,height:19},fill:C.blue,line:{fill:'none',width:0}});
 txt(s,'底盘里程计 / IMU 回传上位机',718,558,496,36,23,C.muted,false,'right');
 txt(s,'供电与安全：电源分路稳压、硬件急停、通信超时停车',66,605,1148,42,25,C.blue,true);
}
// 05 Software/hardware division, native table.
{
 const s=base(5,'软硬件功能划分','模块独立开发，通过 ROS 2 消息与任务接口集成');
 table(s,[['模块','硬件关联','软件功能','输出'],['M1 视觉感知','摄像头 / 昇腾 NPU','人体与物体检测、光斑识别','目标位置与类别'],['M2 语音交互','麦克风 / Atlas CPU','唤醒、流式识别、规则解析','结构化任务指令'],['M3 物品抓取','SO-101 / Atlas','位姿转换、策略推理、臂控制','关节动作与状态'],['M4 导航运动','雷达 / ESP32 / 底盘','SLAM、Nav2、跟随、PID','路径与底盘速度']], [195,275,441,241],64,194,365,24);
 txt(s,'任务管理器统一仲裁跟随、指点导航和抓取，避免控制指令冲突',66,591,1148,55,27,C.blue,true);
}
// 06 Critical issues, horizontal engineering rows.
{
 const s=base(6,'关键技术与难点','先实现可验收基线，再验证学习策略和板端部署');
 const data=[['跟随与避障','视觉识别人，雷达辅助测距','时间同步、坐标标定，加入遮挡恢复与停车'],['地面激光指引','光斑检测与地面平面约束','相机标定求目标坐标，检查地图通行性'],['抓取与昇腾适配','定位误差、策略泛化、算子与延迟','固定作业平面，实测成功率及端侧推理耗时']];
 data.forEach((a,i)=>{const y=194+i*112;txt(s,a[0],66,y,296,48,28,C.navy,true);txt(s,a[1],388,y,820,44,26,C.ink,true);txt(s,a[2],388,y+46,820,40,24,C.muted);rule(s,66,y+100,1148);});
 txt(s,'抓取路线',66,554,210,45,28,C.blue,true);
 const a=box(s,'视觉定位 + 预置动作',292,552,294,61,{size:23});
 const b=box(s,'SmolVLA 微调与适配',644,552,294,61,{size:23});
 connect(s,a,b);
 txt(s,'ACT 备选',966,555,248,48,25,C.muted,true,'center');
 txt(s,'学习策略未达标时保留基线，端侧与外部计算结果分别记录',293,623,920,25,20,C.muted);
}
// 07 Existing work versus next checks.
{
 const s=base(7,'已有基础与待验证内容','已具备软件框架，后续重点转向板端与整机实测');
 txt(s,'已有基础',66,204,545,50,31,C.navy,true);
 txt(s,'待验证内容',686,204,530,50,31,C.navy,true);
 rule(s,640,205,2,C.line,350);
 const left=['整机方案、硬件选型与 ROS 2 接口','语音唤醒、流式 ASR、指令路由代码','规则优先，LLM 兜底与白名单校验','语音模型文件、测试与评测工具'];
 const right=['Atlas 环境、模型转换与推理延迟','真实录音下的唤醒与指令识别率','底盘标定、导航和跟随闭环','抓取成功率与整机供电稳定性'];
 left.forEach((t,i)=>txt(s,t,66,277+i*60,550,44,25,C.ink));
 right.forEach((t,i)=>txt(s,t,686,277+i*60,530,44,25,C.ink));
 txt(s,'25 项通过',66,548,540,70,44,C.blue,true);
 txt(s,'PC 端规则、路由与 LLM 解析相关测试',66,618,1148,30,23,C.muted);
}
// 08 Weekly schedule, native table with every week retained.
{
 const s=base(8,'第 6—14 周进度安排','每周保留可演示成果，第 13 周预留整机联调缓冲');
 table(s,[['周次','主要任务','当周交付'],['第 6 周','Atlas 环境、模型样例、底盘通信与标定','推理样例 + 可控底盘'],['第 7 周','雷达接入、SLAM 建图、Nav2 定点导航','地图 + 导航演示'],['第 8 周','人体检测与雷达测距融合、跟随避障','自主跟随演示'],['第 9 周','地面光斑定位、地图坐标转换与目标导航','激光指引演示'],['第 10 周','语音链路部署、任务解析与调度联调','语音控制整机'],['第 11 周','机械臂标定、抓取基线、示教数据采集','基线抓取 + 数据集'],['第 12 周','SmolVLA / ACT 训练评估、板端适配实测','策略对比 + 路线确定'],['第 13 周','整机联调、故障与供电检查、指标预测试','完整任务闭环'],['第 14 周','统一条件验收、演示视频、报告和代码归档','样机 + 测试记录 + 报告']], [150,670,332],64,187,455,23);
}
// 09 Member ownership, two flat columns.
{
 const s=base(9,'成员分工与协作接口','两人按模块负责，机械结构与整机验收共同完成');
 txt(s,'成员 A',66,201,510,62,40,C.blue,true);
 txt(s,'语音交互与抓取策略',66,276,550,49,30,C.navy,true);
 txt(s,'M2：唤醒、ASR、解析与任务路由',66,346,550,44,25);
 txt(s,'M3：示教采集、模型训练与机械臂控制',66,406,550,44,25);
 txt(s,'负责指令接口与抓取验收记录',66,466,550,44,25,C.muted);
 txt(s,'成员 B',686,201,530,62,40,C.blue,true);
 txt(s,'视觉感知与导航运动',686,276,530,49,30,C.navy,true);
 txt(s,'M1：人体、光斑与物品检测',686,346,530,44,25);
 txt(s,'M4：底盘、SLAM、Nav2 与跟随',686,406,530,44,25);
 txt(s,'负责感知接口与移动验收记录',686,466,530,44,25,C.muted);
 rule(s,66,548,1148);
 txt(s,'共同完成：机械臂安装、供电与急停、接口冻结、整机联调与终验',66,577,1148,48,25,C.blue,true);
}
// 10 Final outcome, no extraneous thank-you page.
{
 const s=base(10,'预期成果','以可重复演示和可追溯测试记录完成课程验收');
 txt(s,'一台机器人样机',66,217,1148,85,56,C.navy,true);
 txt(s,'自主跟随、地面激光指引、语音驱动物品抓取',68,320,1146,60,34,C.blue,true);
 rule(s,66,425,1148);
 txt(s,'交付内容',66,462,230,50,28,C.navy,true);
 txt(s,'ROS 2 软件与配置、演示视频、测试记录、设计报告',324,462,890,50,28);
 txt(s,'验证重点',66,540,230,50,28,C.navy,true);
 txt(s,'成功率、距离误差、定位误差与推理延迟',324,540,890,50,28);
}

await fs.mkdir(path.join(build,'previews'),{recursive:true});
await fs.writeFile(path.join(build,'presentation.json'),JSON.stringify(p.toProto()));
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 await fs.writeFile(path.join(build,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await (await p.export({slide:s,format:'png',scale:1.25})).arrayBuffer()));
 console.log(`Rendered ${i+1}/10`);
}
const candidatePath=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=path.join(out,'昇腾机器人_开题报告.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath,finalPath,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...([3,5,8].flatMap(n=>['--require-native-table-slide',String(n)]))],explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[3,5,8],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify(result));
