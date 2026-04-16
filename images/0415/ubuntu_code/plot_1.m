%% plot_1.m — 读取 thesis_phase_data.txt，绘制论文阶段控制/估计误差（论文导出风格）
%
% 列顺序与 final_0415.py 导出一致（1 行表头）:
%   t_phase_ros_s, ctrl_1_x..ctrl_4_z (12), est_1_x..est_4_z (12)  共 25 列
%
% Gazebo / Ignition 环境刷新率说明（与 mbzirc_ign/worlds 中典型 world 一致）:
%   - 物理步长 physics <max_step_size> 多为 0.004 s → 仿真迭代 250 Hz。
%   - /model/<name>/pose → ROS TF 桥一般随仿真步更新；若 RTF<1，等效 Hz 按比例降低。
%   - 本实验脚本以 ROS 定时器 control_period（默认 0.02 s）读位姿并写盘 → 曲线原始采样约 50 Hz。

close all;

%% 与 picture.m 一致的前 4 机颜色（MATLAB 默认色板）
color = [
    0.0000, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
];

nAgent = 4;

thisdir = fileparts(mfilename('fullpath'));
dataFile = fullfile(thisdir, 'thesis_phase_data.txt');

if ~isfile(dataFile)
    error('找不到数据文件: %s （请先运行 Gazebo 仿真生成 thesis_phase_data.txt）', dataFile);
end

M = readmatrix(dataFile, 'NumHeaderLines', 1);
if size(M, 2) < 25
    error('数据列数不足（期望至少 25 列），当前为 %d 列', size(M, 2));
end

t = M(:, 1);           % 论文阶段相对时间 (s)，自切换控制律起
ctrlBase = 2;          % ctrl_1_x 列（1-based）
estBase = 14;          % est_1_x 列

% %% 平滑窗口（已关闭；需要时可取消注释并改用 ySm 绘图）
% smoothSpanSec = 0.30;
% dtv = diff(t);
% dtv = dtv(isfinite(dtv) & dtv > 1e-9);
% if isempty(dtv)
%     medDt = 0.05;
% else
%     medDt = median(dtv);
% end
% win = max(3, round(smoothSpanSec / medDt));
% if mod(win, 2) == 0
%     win = win + 1;
% end

dims = {'x', 'y', 'z'};
dimIdx = [1, 2, 3];

plotLineWidth = 2.0;
axesFontSize = 13;
labelFontSize = 14;
legendFontSize = 12;
figPos = [80, 80, 620, 420];

legStr = {'UAV 1', 'UAV 2', 'UAV 3', 'UAV 4'};

%% 图 1–3: x / y / z 位置控制误差（四机同图，无子图）
for d = 1:3
    fig = figure( ...
        'Color', 'w', ...
        'Position', figPos, ...
        'Name', sprintf('ctrl_err_%s', dims{d}), ...
        'NumberTitle', 'off');
    clf(fig);
    figure(fig);
    hold on;
    for i = 1:nAgent
        c = ctrlBase + (i - 1) * 3 + (dimIdx(d) - 1);
        yRaw = M(:, c);
        % ySm = movmean(yRaw, win, 'Endpoints', 'shrink');
        plot(t, yRaw, 'LineWidth', plotLineWidth, 'Color', color(i, :));
    end
    applyPubAxesStyle();
    xlabel('$t$ (s)', 'Interpreter', 'latex', 'FontSize', labelFontSize);
    ylabel(sprintf('$\\bar{p}_{%s}$ (m)', dims{d}), 'Interpreter', 'latex', 'FontSize', labelFontSize);
    title(sprintf('Control error, $%s$-axis', dims{d}), ...
        'Interpreter', 'latex', 'FontSize', axesFontSize + 1);
    leg = legend(legStr, 'Location', 'best', 'FontSize', legendFontSize);
    set(leg, 'Box', 'off', 'Color', 'none', 'FontName', 'Times New Roman');
    xlim([min(t), max(t)]);
    hold off;
end

%% 图 4–6: x / y / z 估计误差
for d = 1:3
    fig = figure( ...
        'Color', 'w', ...
        'Position', figPos, ...
        'Name', sprintf('est_err_%s', dims{d}), ...
        'NumberTitle', 'off');
    clf(fig);
    figure(fig);
    hold on;
    for i = 1:nAgent
        c = estBase + (i - 1) * 3 + (dimIdx(d) - 1);
        yRaw = M(:, c);
        % ySm = movmean(yRaw, win, 'Endpoints', 'shrink');
        plot(t, yRaw, 'LineWidth', plotLineWidth, 'Color', color(i, :));
    end
    applyPubAxesStyle();
    xlabel('$t$ (s)', 'Interpreter', 'latex', 'FontSize', labelFontSize);
    ylabel(sprintf('$\\tilde{p}_{%s}$ (m)', dims{d}), 'Interpreter', 'latex', 'FontSize', labelFontSize);
    title(sprintf('Estimation error, $%s$-axis', dims{d}), ...
        'Interpreter', 'latex', 'FontSize', axesFontSize + 1);
    leg = legend(legStr, 'Location', 'best', 'FontSize', legendFontSize);
    set(leg, 'Box', 'off', 'Color', 'none', 'FontName', 'Times New Roman');
    xlim([min(t), max(t)]);
    hold off;
end

function applyPubAxesStyle()
    fig = gcf;
    set(fig, 'Color', 'w', 'InvertHardcopy', 'off', 'PaperPositionMode', 'auto');
    ax = gca;
    set(ax, ...
        'Color', 'w', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 13, ...
        'LineWidth', 1.05, ...
        'Box', 'on', ...
        'TickDir', 'out', ...
        'TickLength', [0.018, 0.018], ...
        'XMinorTick', 'on', ...
        'YMinorTick', 'on', ...
        'GridAlpha', 0.45, ...
        'GridColor', [0.55 0.55 0.55], ...
        'MinorGridAlpha', 0.12, ...
        'MinorGridColor', [0.75 0.75 0.75], ...
        'MinorGridLineStyle', ':');
    grid(ax, 'on');
    grid(ax, 'minor');
end
