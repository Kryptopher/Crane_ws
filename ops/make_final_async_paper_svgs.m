%% make_final_async_paper_svgs.m
% Generate publication-ready figures for the Final Async
% 1000 mm / 150 mm/s gantry experiments.
%
% Every figure is exported as:
%   PDF - primary vector format for LaTeX/Overleaf
%   PNG - 600 dpi fallback
%   SVG - editable secondary copy
%
% Outputs are written to:
%   <BASE_DIR>\Figures_FinalAsync_SVG

clear; clc; close all;
set(0, "DefaultFigureVisible", "off");

%% ---------------- USER SETTINGS ----------------
BASE_DIR = "C:\Users\sanja\OneDrive - Louisiana State University\Sanjay_Maharjan\02_Research\Project_05_Camera_Crane\Experiment\Final_Async_1000mm_v150_20260622";

LENGTH_ORDER = [0.50 0.70 0.90 1.10 1.30];
METHOD_ORDER = ["Pulse", "Nonrobust", "Robust", "AIS"];
METHOD_COLOR = containers.Map( ...
    {'Pulse','Nonrobust','Robust','AIS'}, ...
    {[0.55 0.58 0.62], [0.25 0.48 0.90], [0.90 0.20 0.18], [0.18 0.65 0.30]});
METHOD_OFFSET = containers.Map( ...
    {'Pulse','Nonrobust','Robust','AIS'}, ...
    {-0.045, -0.015, 0.015, 0.045});

REPRESENTATIVE_L_M = 1.30;
REPRESENTATIVE_REP = 5;
TAU_S = 3.7;
TARGET_MM = 1000.0;
VMAX_MM_S = 150.0;

OUT_DIR = fullfile(BASE_DIR, "Figures_FinalAsync_SVG");
if ~exist(OUT_DIR, "dir"); mkdir(OUT_DIR); end
staleFiles = ["residual_angle_boxplots.svg", "modal_decomposition_example.svg"];
for i = 1:numel(staleFiles)
    stalePath = fullfile(OUT_DIR, staleFiles(i));
    if exist(stalePath, "file"); delete(stalePath); end
end
staleModalBase = fullfile(OUT_DIR, "modal_decomposition_nonrobust_L090_rep01");
for ext = [".svg", ".pdf", ".png"]
    stalePath = staleModalBase + ext;
    if exist(stalePath, "file"); delete(stalePath); end
end
REPORT_FILE = fullfile(OUT_DIR, "paper_figure_statistics_report.txt");
% A previous failed run may have left MATLAB's diary file open.
diary off;
if exist(REPORT_FILE, "file"); delete(REPORT_FILE); end
diary(REPORT_FILE);

MODAL_DIR = fullfile(BASE_DIR, "modal_final_v3_quant_filtered");
MODAL_CSV = fullfile(MODAL_DIR, "modal_10s_results_gates_bestof10.csv");
BESTFIT_DIR = fullfile(MODAL_DIR, "best_fit_decompositions");

fprintf("BASE_DIR:\n%s\n", BASE_DIR);
fprintf("OUT_DIR:\n%s\n\n", OUT_DIR);

%% ---------------- DISCOVER RUNS ----------------
runs = discoverRuns(BASE_DIR, LENGTH_ORDER);
fprintf("Discovered %d raw.csv files.\n", height(runs));
if isempty(runs)
    error("No raw.csv files found. Check BASE_DIR.");
end
printRunInventory(runs, LENGTH_ORDER, METHOD_ORDER);

%% ---------------- TIME-HISTORY FIGURES ----------------
for im = 1:numel(METHOD_ORDER)
    method = METHOD_ORDER(im);
    row = selectRepresentativeRun(runs, REPRESENTATIVE_L_M, method, REPRESENTATIVE_REP);
    if isempty(row)
        warning("No representative run found for %s at L=%.2f.", method, REPRESENTATIVE_L_M);
        continue;
    end
    data = loadRawFlexible(row.raw_file);
    outSvg = fullfile(OUT_DIR, sprintf("timeseries_%s_L%03d.svg", lowerMethodTag(method), round(100*REPRESENTATIVE_L_M)));
    makeTimeHistoryFigure(data, row, method, METHOD_COLOR(char(method)), ...
        TAU_S, TARGET_MM, VMAX_MM_S, outSvg);
end

%% ---------------- ONLINE ID FIGURE ----------------
outId = fullfile(OUT_DIR, "online_zero_zeta_id_convergence.svg");
makeOnlineIdFigure(runs, LENGTH_ORDER, TAU_S, outId);

%% ---------------- RESIDUAL ANGLE FIGURES FROM raw.csv ----------------
makeResidualAnglePlots(runs, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, OUT_DIR);

%% ---------------- MODAL ENERGY FIGURES ----------------
if exist(MODAL_CSV, "file")
    modal = readtable(MODAL_CSV, "VariableNamingRule", "preserve");
    modal.method = normalizeMethodStrings(string(modal.method));

    modal = modal(abs(modal.gate_after_stop_s - 0.20) < 1e-9, :);
    printAndSaveEnergyStatistics(modal, LENGTH_ORDER, METHOD_ORDER, OUT_DIR);
    printModalExampleStatistics(modal);

    outEnergy = fullfile(OUT_DIR, "residual_energy_primary.svg");
    makePrimaryEnergyFigure(modal, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, outEnergy);

    outReduction = fullfile(OUT_DIR, "residual_energy_reduction_vs_pulse.svg");
    makeEnergyReductionFigure(modal, LENGTH_ORDER, METHOD_ORDER, METHOD_COLOR, outReduction);

    outModalPulse = fullfile(OUT_DIR, "modal_decomposition_pulse_L130_rep01.svg");
    makeModalDecompositionExample(BESTFIT_DIR, 1.30, "Pulse", 1, outModalPulse);
else
    warning("Modal CSV not found:\n%s\nSkipping modal energy figures.", MODAL_CSV);
end

fprintf("\nDONE. Figures saved to:\n%s\n", OUT_DIR);
fprintf("Paper statistics report saved to:\n%s\n", REPORT_FILE);
diary off;
set(0, "DefaultFigureVisible", "on");

%% ========================================================================
% Local functions
%% ========================================================================

function runs = discoverRuns(BASE_DIR, LENGTH_ORDER)
    rows = table();
    for iL = 1:numel(LENGTH_ORDER)
        L = LENGTH_ORDER(iL);
        Lfolder = sprintf("L%03d", round(100*L));
        thisRoot = fullfile(BASE_DIR, Lfolder);
        d = dir(fullfile(thisRoot, "**", "raw.csv"));
        for k = 1:numel(d)
            rawFile = string(fullfile(d(k).folder, d(k).name));
            info = parseRunInfo(rawFile);
            row = table();
            row.raw_file = rawFile;
            row.run_folder = string(d(k).folder);
            row.run_name = string(getLastPathPart(d(k).folder));
            row.length_m = info.length_m;
            row.method = info.method;
            row.rep = info.rep;
            row.summary_file = string(fullfile(d(k).folder, "summary.txt"));
            rows = [rows; row]; %#ok<AGROW>
        end
    end
    if isempty(rows)
        runs = table();
    else
        rows = sortrows(rows, ["length_m", "method", "rep", "run_name"]);
        G = findgroups(rows.length_m, rows.method, rows.rep);
        keep = splitapply(@(idx) idx(end), (1:height(rows))', G);
        if numel(keep) < height(rows)
            warning("Found %d duplicate length/method/rep runs; using the latest run name in each group.", height(rows)-numel(keep));
        end
        runs = rows(sort(keep), :);
    end
end

function info = parseRunInfo(rawFile)
    s = lower(string(rawFile));
    tokPhys = regexp(s, "physical_l(\d{3})", "tokens", "once");
    tokFolder = regexp(s, "[\\/]+l(\d{3})[\\/]+", "tokens", "once");
    tokAny = regexp(s, "l(\d{3})", "tokens", "once");
    if ~isempty(tokPhys)
        info.length_m = str2double(tokPhys{1}) / 100;
    elseif ~isempty(tokFolder)
        info.length_m = str2double(tokFolder{1}) / 100;
    elseif ~isempty(tokAny)
        info.length_m = str2double(tokAny{1}) / 100;
    else
        info.length_m = NaN;
    end

    if contains(s, "pulse")
        info.method = "Pulse";
    elseif contains(s, "nonrobust") || contains(s, "_zv_")
        info.method = "Nonrobust";
    elseif contains(s, "robust")
        info.method = "Robust";
    elseif contains(s, "zerozeta") || contains(s, "zero_zeta") || contains(s, "is2") || contains(s, "isa")
        info.method = "AIS";
    else
        info.method = "Unknown";
    end

    tokRep = regexp(s, "rep(\d+)", "tokens", "once");
    if ~isempty(tokRep)
        info.rep = str2double(tokRep{1});
    else
        info.rep = NaN;
    end
end

function row = selectRepresentativeRun(runs, L, method, preferredRep)
    idx = abs(runs.length_m - L) < 1e-9 & runs.method == method;
    candidates = runs(idx, :);
    if isempty(candidates)
        row = table();
        return;
    end
    idxRep = candidates.rep == preferredRep;
    if any(idxRep)
        candidates = candidates(idxRep, :);
    end
    row = candidates(1, :);
end

function data = loadRawFlexible(rawFile)
    opts = detectImportOptions(rawFile, "FileType", "text");
    opts.VariableNamingRule = "preserve";
    T = readtable(rawFile, opts);
    names = string(T.Properties.VariableNames);

    data.raw_file = string(rawFile);
    data.t = getNum(T, names, ["move_time_sec", "t_motion_sec", "time_sec", "t"]);
    data.wall_time = getNumOpt(T, names, ["wall_time_sec", "stamp_sec"]);
    data.cmd_vx = getNumOpt(T, names, ["cmd_vx_mm_s", "vx_cmd_mm_s", "cmd_vel_x_mm_s"]);
    data.cart_vx = getNumOpt(T, names, ["cart_vx_mm_s", "vx_mm_s", "gantry_vx_mm_s"]);
    data.cart_q = getNumOpt(T, names, ["cart_q_mm", "cart_x_mm", "q_mm"]);
    data.swing_mm = getNumOpt(T, names, ["swing_mm", "payload_rel_mm", "payload_x_rel_mm"]);
    data.swing_angle_deg = getNumOpt(T, names, ["swing_axis_angle_deg", "swing_pitch_deg", "enc_pitch_deg", "payload_pitch_deg"]);
    data.gantry_state_age_ms = getNumOpt(T, names, ["gantry_state_age_ms"]);
    data.payload_wall_age_ms = getNumOpt(T, names, ["payload_wall_age_ms"]);
    data.stream_dt_ms = getNumOpt(T, names, ["stream_dt_ms"]);
    data.zero_zeta_T = getNumOpt(T, names, ["zero_zeta_T_sec"]);
    data.zero_zeta_score = getNumOpt(T, names, ["zero_zeta_score"]);
    data.schedule_T = getNumOpt(T, names, ["schedule_T_sec"]);
    data.schedule_locked_at = getNumOpt(T, names, ["schedule_locked_at_s"]);
    data.schedule_id_time = getNumOpt(T, names, ["schedule_id_time_s"]);
    data.schedule_id_T = getNumOpt(T, names, ["schedule_id_T_sec"]);
    data.id_candidate_T = getNumOpt(T, names, ["id_candidate_T_sec", "T_sec"]);
    data.id_candidate_valid = getNumOpt(T, names, ["id_candidate_valid"]);

    n = numel(data.t);
    fns = fieldnames(data);
    for i = 1:numel(fns)
        f = fns{i};
        if isnumeric(data.(f)) && numel(data.(f)) ~= n
            data.(f) = resizeToN(data.(f), n);
        end
    end

    valid = isfinite(data.t);
    data = subsetData(data, valid);
    if ~isempty(data.t)
        data.t = data.t - data.t(1);
    end
end

function data = subsetData(data, idx)
    fns = fieldnames(data);
    for i = 1:numel(fns)
        f = fns{i};
        x = data.(f);
        if isnumeric(x) && numel(x) == numel(idx)
            data.(f) = x(idx);
        end
    end
end

function makeTimeHistoryFigure(data, row, method, methodColor, tauS, targetMm, vmaxMmS, outSvg)
    t = data.t(:);
    cmd = data.cmd_vx(:);
    act = data.cart_vx(:);
    swingDeg = data.swing_angle_deg(:);
    swingMm = data.swing_mm(:);
    tf = targetMm / vmaxMmS;
    stopT = detectStopTime(t, cmd);
    switches = detectSwitchTimes(t, cmd);
    lockT = firstFiniteScalar(data.schedule_locked_at);
    schedT = firstFiniteScalar(data.schedule_T);

    % Online-ID convergence has its own figure, so all method histories use
    % the same two-panel velocity/swing layout.
    nTiles = 2;
    figHeight = 1020;
    fig = figure("Color", "w", "Position", [80 80 1450 figHeight]);
    tlo = tiledlayout(fig, nTiles, 1, "TileSpacing", "compact", "Padding", "compact");

    ax1 = nexttile(tlo); hold(ax1, "on"); grid(ax1, "on");
    hCmd = plot(ax1, t, cmd, "--", "Color", [0.15 0.15 0.15], ...
        "LineWidth", 1.7, "DisplayName", "Commanded cart velocity");
    hAct = gobjects(0);
    if any(isfinite(act))
        hAct = plot(ax1, t, act, "Color", methodColor, "LineWidth", 1.4, ...
            "DisplayName", "Measured cart velocity");
    end
    if method == "AIS"
        addTauLine(ax1, tauS);
    end
    ylabel(ax1, "Cart velocity (mm/s)");

    ax2 = nexttile(tlo); hold(ax2, "on"); grid(ax2, "on");
    if any(isfinite(swingDeg))
        hSwing = plot(ax2, t, swingDeg, "Color", methodColor, "LineWidth", 1.3, ...
            "DisplayName", "Payload swing angle");
        ylabel(ax2, "Swing angle (deg)");
    elseif any(isfinite(swingMm))
        hSwing = plot(ax2, t, swingMm, "Color", methodColor, "LineWidth", 1.3, ...
            "DisplayName", "Payload swing");
        ylabel(ax2, "Payload swing (mm)");
    else
        hSwing = gobjects(0);
    end
    if method == "AIS"
        addTauLine(ax2, tauS);
    end

    % Use one untitled legend per exported method figure. A proxy on the
    % velocity axes represents the swing trace from the second tile.
    legendHandles = hCmd;
    legendLabels = "Commanded cart velocity";
    if ~isempty(hAct)
        legendHandles(end+1) = hAct; %#ok<AGROW>
        legendLabels(end+1) = "Measured cart velocity"; %#ok<AGROW>
    end
    if ~isempty(hSwing)
        hSwingProxy = plot(ax1, nan, nan, "-", "Color", methodColor, ...
            "LineWidth", 1.3, "DisplayName", "Payload swing angle");
        legendHandles(end+1) = hSwingProxy; %#ok<AGROW>
        legendLabels(end+1) = "Payload swing angle"; %#ok<AGROW>
    end
    legend(ax1, legendHandles, legendLabels, "Location", "northoutside", ...
        "Orientation", "horizontal", "NumColumns", numel(legendHandles), ...
        "Box", "off");

    axesToLink = [ax1 ax2];
    ax1.XTickLabel = [];
    xlabel(tlo, "$t~(\mathrm{s})$", "Interpreter", "latex", ...
        "FontName", "Times New Roman", "FontSize", 15);
    linkaxes(axesToLink, "x");
    % Use the same time range in all four files so the 2-by-2 LaTeX
    % composition preserves a meaningful timing comparison.
    tFinite = t(isfinite(t));
    if ~isempty(tFinite)
        xlim(ax1, [0, min(max(tFinite), 15.0)]);
    end

    moveMask = isfinite(t) & isfinite(cmd) & isfinite(act) & t <= stopT;
    residualMask = isfinite(t) & isfinite(swingDeg) & ...
        t >= stopT + 0.20 & t <= stopT + 10.20;
    velRmse = NaN; swingRms = NaN; swingP2P = NaN;
    travelMm = NaN;
    if any(moveMask); velRmse = sqrt(mean((act(moveMask)-cmd(moveMask)).^2)); end
    qMask = isfinite(t) & isfinite(data.cart_q) & t <= stopT;
    if nnz(qMask) >= 2
        q = data.cart_q(qMask); travelMm = q(end)-q(1);
    end
    if any(residualMask)
        sr = swingDeg(residualMask);
        swingRms = sqrt(mean((sr-mean(sr)).^2));
        swingP2P = max(sr)-min(sr);
    end
    fprintf("REPRESENTATIVE %-10s L=%.2f rep%02d | stop=%.3f s | travel=%.3f mm | velocity RMSE=%.3f mm/s | residual RMS=%.5f deg | residual p2p=%.5f deg\n", ...
        method, row.length_m, row.rep, stopT, travelMm, velRmse, swingRms, swingP2P);
    fprintf("  command switches [s]:"); fprintf(" %.3f", switches); fprintf("\n");
    if method == "AIS"
        selectedTosc = 2 * firstFiniteScalar(data.schedule_id_T);
        fprintf("  selected ID: t=%.3f s, Tosc=%.5f s | tau/lock=%.3f s | closed-form schedule delay=%.5f s\n", ...
            firstFiniteScalar(data.schedule_id_time), selectedTosc, ...
            lockT, schedT);
    end

    exportSvg(fig, outSvg);
    close(fig);
    fprintf("Saved %s\n", outSvg);
end

function makeOnlineIdFigure(runs, LENGTH_ORDER, tauS, outSvg)
    aisRuns = runs(runs.method == "AIS", :);
    fig = figure("Color", "w", "Position", [80 80 1700 1100]);
    tlo = tiledlayout(fig, numel(LENGTH_ORDER), 1, "TileSpacing", "compact", "Padding", "compact");
    g = 9.81;
    aisColor = [0.18 0.65 0.30];
    lengthColors = [
        aisColor + 0.55*(1-aisColor)
        aisColor + 0.28*(1-aisColor)
        aisColor
        0.78*aisColor
        0.58*aisColor
    ];
    idRows = table();
    idAxes = gobjects(numel(LENGTH_ORDER), 1);

    for iL = 1:numel(LENGTH_ORDER)
        L = LENGTH_ORDER(iL);
        ax = nexttile(tlo); hold(ax, "on"); grid(ax, "on");
        idAxes(iL) = ax;
        idx = abs(aisRuns.length_m - L) < 1e-9;
        these = aisRuns(idx, :);
        loaded = cell(height(these), 1);
        selectedHalfPeriods = nan(height(these), 1);

        % Preserve statistics from all repetitions, but display only the
        % middle run after sorting by the selected period.
        for r = 1:height(these)
            data = loadRawFlexible(these.raw_file(r));
            loaded{r} = data;
            selectedTime = firstFiniteScalar(data.schedule_id_time);
            selectedHalfPeriod = firstFiniteScalar(data.schedule_id_T);
            lockTime = firstFiniteScalar(data.schedule_locked_at);
            selectedHalfPeriods(r) = selectedHalfPeriod;
            if isfinite(selectedTime) && isfinite(selectedHalfPeriod)
                idRow = table();
                idRow.length_m = L;
                idRow.rep = these.rep(r);
                idRow.selected_id_time_s = selectedTime;
                idRow.selected_id_Tosc_s = 2*selectedHalfPeriod;
                idRow.lock_time_s = lockTime;
                idRow.reference_Tosc_s = 2*pi/sqrt(g/L);
                idRow.error_pct = 100 * ...
                    (idRow.selected_id_Tosc_s-idRow.reference_Tosc_s) / ...
                    idRow.reference_Tosc_s;
                idRows = [idRows; idRow]; %#ok<AGROW>
            end
        end

        validRuns = find(isfinite(selectedHalfPeriods));
        validRuns = validRuns(:);
        if ~isempty(validRuns)
            % Use a numeric matrix here so row/column orientation and
            % table name-value syntax are independent of MATLAB release.
            orderValues = [selectedHalfPeriods(validRuns), ...
                reshape(these.rep(validRuns), [], 1), validRuns];
            orderValues = sortrows(orderValues, [1 2]);
            middleRun = orderValues(ceil(size(orderValues,1)/2), 3);
        else
            middleRun = find(cellfun(@(d) any(isfinite(d.zero_zeta_T)), loaded), 1, "first");
        end

        if ~isempty(middleRun)
            data = loaded{middleRun};
            Tosc = 2*data.zero_zeta_T;
            valid = isfinite(data.t) & isfinite(Tosc);
            plot(ax, data.t(valid), Tosc(valid), "-", ...
                "Color", lengthColors(iL,:), "LineWidth", 2.0);
            fprintf("SYSTEM ID representative L=%.2f m: rep%02d, selected Tosc=%.5f s\n", ...
                L, these.rep(middleRun), 2*selectedHalfPeriods(middleRun));
        else
            warning("No valid AIS identification trace for L=%.2f m.", L);
        end

        referenceTosc = 2*pi/sqrt(g/L);
        yline(ax, referenceTosc, ":", "$T_{\mathrm{osc}}$", ...
            "Interpreter", "latex", "Color", [0.28 0.28 0.28], ...
            "LineWidth", 1.3, "FontSize", 9, ...
            "LabelHorizontalAlignment", "right", ...
            "LabelVerticalAlignment", "bottom", ...
            "HandleVisibility", "off");
        text(ax, 0.018, 0.82, sprintf("$L=%.2f~\\mathrm{m}$", L), ...
            "Units", "normalized", "Interpreter", "latex", ...
            "FontName", "Times New Roman", "FontSize", 10, ...
            "FontWeight", "bold", "VerticalAlignment", "top");
        xlim(ax, [0.5, 4.0]);
        ylim(ax, [0.70, 3.10]);
        if iL < numel(LENGTH_ORDER)
            ax.XTickLabel = [];
        end
    end

    linkaxes(idAxes, "x");
    xlabel(tlo, "$t~(\mathrm{s})$", "Interpreter", "latex", ...
        "FontName", "Times New Roman", "FontSize", 11.5);
    ylabel(tlo, "$\widehat{T}_{\mathrm{osc}}~(\mathrm{s})$", ...
        "Interpreter", "latex", "FontName", "Times New Roman", ...
        "FontSize", 11);
    title(tlo, "System ID", ...
        "FontSize", 13, "FontWeight", "bold");
    drawnow;
    addSharedTauLine(fig, idAxes, tauS);
    printAndSaveIdStatistics(idRows, fileparts(outSvg));
    exportSvg(fig, outSvg);
    close(fig);
    fprintf("Saved %s\n", outSvg);
end

function makeResidualAnglePlots(runs, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, OUT_DIR)
    rows = table();
    for i = 1:height(runs)
        data = loadRawFlexible(runs.raw_file(i));
        stopT = detectStopTime(data.t, data.cmd_vx);
        angle = data.swing_angle_deg(:);
        mask = isfinite(data.t) & isfinite(angle) & ...
            data.t >= stopT + 0.20 & data.t <= stopT + 10.20;
        residual = angle(mask);
        row = table();
        row.length_m = runs.length_m(i);
        row.method = runs.method(i);
        row.rep = runs.rep(i);
        if numel(residual) >= 2
            row.rms_demeaned_deg = sqrt(mean((residual - mean(residual)).^2));
            row.p2p_deg = max(residual) - min(residual);
            row.rms_deg = sqrt(mean(residual.^2));
        else
            row.rms_demeaned_deg = NaN;
            row.p2p_deg = NaN;
            row.rms_deg = NaN;
        end
        rows = [rows; row]; %#ok<AGROW>
    end
    printAndSaveAngleStatistics(rows, LENGTH_ORDER, METHOD_ORDER, OUT_DIR);

    metrics = {
        "rms_demeaned_deg", "$\theta_{\mathrm{RMS}}~(\mathrm{deg})$", "residual_angle_rms.svg"
        "p2p_deg", "$\theta_{\mathrm{p-p}}~(\mathrm{deg})$", "residual_angle_p2p.svg"
        "rms_deg", "$\theta_{\mathrm{RMS,offset}}~(\mathrm{deg})$", "residual_angle_rms_with_offset.svg"
    };
    for i = 1:size(metrics,1)
        makeSingleMetricBoxplot(rows, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, ...
            metrics{i,1}, metrics{i,2}, fullfile(OUT_DIR, metrics{i,3}));
    end
end

function makeSingleMetricBoxplot(T, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, metric, yLabel, outSvg)
    fig = figure("Color", "w", "Position", [80 80 1420 740]);
    ax = axes(fig); hold(ax, "on"); grid(ax, "on"); box(ax, "on");
    rng(31);
    hLegend = gobjects(numel(METHOD_ORDER),1);
    for im = 1:numel(METHOD_ORDER)
        method = METHOD_ORDER(im);
        idx = T.method == method;
        x = T.length_m(idx) + METHOD_OFFSET(char(method));
        y = T.(metric)(idx);
        valid = isfinite(x) & isfinite(y) & y > 0;
        c = METHOD_COLOR(char(method));
        if any(valid)
            boxchart(ax, x(valid), y(valid), "BoxFaceColor", c, "BoxFaceAlpha", 0.38, ...
                "WhiskerLineColor", c, "MarkerStyle", "none", "BoxWidth", 0.027);
            scatter(ax, x(valid)+(rand(nnz(valid),1)-0.5)*0.008, y(valid), 31, ...
                "MarkerFaceColor", c, "MarkerEdgeColor", [0.15 0.15 0.15], ...
                "MarkerFaceAlpha", 0.82, "LineWidth", 0.45);
        end
        hLegend(im) = plot(ax, nan, nan, "s", "MarkerFaceColor", c, ...
            "MarkerEdgeColor", c, "MarkerSize", 8, "DisplayName", method);
    end
    xline(ax, 0.90, "k--", "Nominal model", "LineWidth", 1.0, "HandleVisibility", "off");
    set(ax, "YScale", "log", "FontName", "Times New Roman", "FontSize", 13, "LineWidth", 0.8);
    xticks(ax, LENGTH_ORDER); xticklabels(ax, compose("%.1f", LENGTH_ORDER));
    xlim(ax, [0.41 1.39]);
    xlabel(ax, "$L~(\mathrm{m})$", "Interpreter", "latex");
    ylabel(ax, yLabel, "Interpreter", "latex");
    lgd = legend(ax, hLegend, METHOD_ORDER, "Location", "northoutside", ...
        "Orientation", "horizontal", "NumColumns", 4, "Box", "off");
    exportSvg(fig, outSvg); close(fig);
    fprintf("Saved %s\n", outSvg);
end

function makeResidualEnergyBoxplots(modal, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, outPng)
    metrics = {
        "mode1_energy", "Mode 1 energy [J s/kg]"
        "mode2_energy", "Mode 2 energy [J s/kg]"
        "measured_total_energy", "Measured residual energy [J s/kg]"
        "A1_abs_deg", "Mode 1 amplitude [deg]"
        "A2_abs_deg", "Mode 2 amplitude [deg]"
        "energy_explained_pct", "Energy explained [%]"
    };
    makeMetricBoxplotGrid(modal, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, metrics, "Residual modal metrics, 0.20 s guard", outPng, true);
end

function makePrimaryEnergyFigure(modal, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, outSvg)
    metric = "mode1_Eavg_reported_J_per_kg";
    requireColumn(modal, metric);

    fig = figure("Color", "w", "Position", [80 80 1450 760]);
    set(fig, "Renderer", "painters");
    ax = axes(fig); hold(ax, "on"); grid(ax, "on"); box(ax, "on");
    rng(22);

    legendHandles = gobjects(numel(METHOD_ORDER), 1);
    for im = 1:numel(METHOD_ORDER)
        method = METHOD_ORDER(im);
        idx = modal.method == method;
        x = modal.length_m(idx) + METHOD_OFFSET(char(method));
        y = modal.(metric)(idx);
        valid = isfinite(x) & isfinite(y) & y > 0;
        c = METHOD_COLOR(char(method));
        if any(valid)
            boxchart(ax, x(valid), y(valid), "BoxFaceColor", c, ...
                "BoxFaceAlpha", 0.38, "WhiskerLineColor", c, ...
                "MarkerStyle", "none", "BoxWidth", 0.027);
            scatter(ax, x(valid) + (rand(nnz(valid),1)-0.5)*0.008, y(valid), 31, ...
                "MarkerFaceColor", c, "MarkerEdgeColor", [0.15 0.15 0.15], ...
                "MarkerFaceAlpha", 0.82, "LineWidth", 0.45);
        end
        legendHandles(im) = plot(ax, nan, nan, "s", "MarkerFaceColor", c, ...
            "MarkerEdgeColor", c, "MarkerSize", 8, "DisplayName", method);
    end

    xline(ax, 0.90, "k--", "Nominal model", "LineWidth", 1.1, ...
        "LabelVerticalAlignment", "top", "LabelOrientation", "horizontal", ...
        "FontSize", 9);
    set(ax, "YScale", "log", "FontName", "Times New Roman", "FontSize", 13, ...
        "LineWidth", 0.8, "Layer", "top");
    xticks(ax, LENGTH_ORDER); xticklabels(ax, compose("%.1f", LENGTH_ORDER));
    xlim(ax, [0.41 1.39]);
    xlabel(ax, "$L~(\mathrm{m})$", "Interpreter", "latex");
    ylabel(ax, "$\overline{e}_1~(\mathrm{J\,kg^{-1}})$", ...
        "Interpreter", "latex");
    lgd = legend(ax, legendHandles, METHOD_ORDER, "Location", "northoutside", ...
        "Orientation", "horizontal", "NumColumns", 4, "Box", "off");
    exportSvg(fig, outSvg); close(fig);
    fprintf("Saved %s\n", outSvg);
end

function makeEnergyReductionFigure(modal, LENGTH_ORDER, METHOD_ORDER, METHOD_COLOR, outSvg)
    metric = "mode1_Eavg_reported_J_per_kg";
    requireColumn(modal, metric);
    shaped = METHOD_ORDER(METHOD_ORDER ~= "Pulse");
    reduction = nan(numel(LENGTH_ORDER), numel(shaped));

    for iL = 1:numel(LENGTH_ORDER)
        L = LENGTH_ORDER(iL);
        pulse = modal.(metric)(abs(modal.length_m-L)<1e-9 & modal.method=="Pulse");
        pulseMean = mean(pulse(isfinite(pulse)));
        for im = 1:numel(shaped)
            vals = modal.(metric)(abs(modal.length_m-L)<1e-9 & modal.method==shaped(im));
            shapedMean = mean(vals(isfinite(vals)));
            reduction(iL, im) = 100 * (1 - shapedMean / pulseMean);
        end
    end

    fig = figure("Color", "w", "Position", [80 80 1450 720]);
    set(fig, "Renderer", "painters");
    ax = axes(fig); hold(ax, "on"); grid(ax, "on"); box(ax, "on");
    b = bar(ax, LENGTH_ORDER, reduction, "grouped", "BarWidth", 0.78);
    for im = 1:numel(shaped)
        b(im).FaceColor = METHOD_COLOR(char(shaped(im)));
        b(im).EdgeColor = "none";
        b(im).DisplayName = shaped(im);
    end
    yline(ax, 99, "k:", "99%", "LineWidth", 1.0, "LabelHorizontalAlignment", "left");
    set(ax, "FontName", "Times New Roman", "FontSize", 13, "LineWidth", 0.8, "Layer", "top");
    xticks(ax, LENGTH_ORDER); xticklabels(ax, compose("%.1f", LENGTH_ORDER));
    xlim(ax, [0.40 1.40]); ylim(ax, [75 100.5]);
    xlabel(ax, "$L~(\mathrm{m})$", "Interpreter", "latex");
    ylabel(ax, "Reduction relative to pulse (\%)", "Interpreter", "latex");
    lgd = legend(ax, "Location", "northoutside", "Orientation", "horizontal", ...
        "NumColumns", 3, "Box", "off");
    exportSvg(fig, outSvg); close(fig);
    fprintf("Saved %s\n", outSvg);
end

function makeMetricBoxplotGrid(T, LENGTH_ORDER, METHOD_ORDER, METHOD_OFFSET, METHOD_COLOR, metrics, titleText, outSvg, logEnergy)
    fig = figure("Color", "w", "Position", [80 80 1750 1050]);
    tlo = tiledlayout(fig, 2, 3, "TileSpacing", "compact", "Padding", "compact");
    rng(7);
    for iM = 1:size(metrics, 1)
        metric = metrics{iM, 1};
        label = metrics{iM, 2};
        ax = nexttile(tlo); hold(ax, "on"); grid(ax, "on");
        for im = 1:numel(METHOD_ORDER)
            method = METHOD_ORDER(im);
            idx = T.method == method;
            if ~any(idx) || ~ismember(metric, string(T.Properties.VariableNames)); continue; end
            x = T.length_m(idx);
            y = T.(metric)(idx);
            x = x + METHOD_OFFSET(char(method));
            c = METHOD_COLOR(char(method));
            valid = isfinite(x) & isfinite(y);
            if ~any(valid); continue; end
            boxchart(ax, x(valid), y(valid), "BoxFaceColor", c, "BoxFaceAlpha", 0.35, ...
                "WhiskerLineColor", c, "MarkerStyle", "none", "BoxWidth", 0.030);
            scatter(ax, x(valid) + (rand(nnz(valid),1)-0.5)*0.010, y(valid), 24, ...
                "MarkerFaceColor", c, "MarkerEdgeColor", [0.15 0.15 0.15], ...
                "MarkerFaceAlpha", 0.75, "MarkerEdgeAlpha", 0.5);
        end
        xline(ax, 0.90, "k--", "nominal", "LineWidth", 1.0);
        xlabel(ax, "Rope length [m]"); ylabel(ax, label);
        title(ax, label, "Interpreter", "none");
        xticks(ax, LENGTH_ORDER); xticklabels(ax, compose("%.2f", LENGTH_ORDER));
        xlim(ax, [min(LENGTH_ORDER)-0.12, max(LENGTH_ORDER)+0.12]);
        if logEnergy && contains(metric, "energy")
            vals = T.(metric);
            vals = vals(isfinite(vals) & vals > 0);
            if ~isempty(vals) && max(vals)/max(min(vals), eps) > 30
                set(ax, "YScale", "log");
            end
        end
    end
    ax = nexttile(tlo, 1); hold(ax, "on");
    h = gobjects(numel(METHOD_ORDER), 1);
    for im = 1:numel(METHOD_ORDER)
        method = METHOD_ORDER(im);
        c = METHOD_COLOR(char(method));
        h(im) = plot(ax, nan, nan, "s", "MarkerFaceColor", c, "MarkerEdgeColor", c, "DisplayName", method);
    end
    legend(ax, h, METHOD_ORDER, "Orientation", "horizontal", "Location", "northoutside");
    title(tlo, titleText, "FontSize", 15, "FontWeight", "bold");
    exportSvg(fig, outSvg);
    close(fig);
    fprintf("Saved %s\n", outSvg);
end

function makeModalDecompositionExample(BESTFIT_DIR, L, method, rep, outSvg)
    if ~exist(BESTFIT_DIR, "dir")
        warning("Best-fit directory missing:\n%s", BESTFIT_DIR);
        return;
    end
    if method == "AIS"
        methodTag = "Zero-zeta_ISA";
    else
        methodTag = method;
    end
    pattern = sprintf("bestfit_*_L%03d_%s_rep%02d_*.csv", round(100*L), methodTag, rep);
    d = dir(fullfile(BESTFIT_DIR, pattern));
    if isempty(d)
        d = dir(fullfile(BESTFIT_DIR, sprintf("bestfit_*_L%03d_*%s*.csv", round(100*L), methodTag)));
    end
    if isempty(d)
        warning("No modal decomposition CSV found for L=%.2f method=%s.", L, method);
        return;
    end
    T = readtable(fullfile(d(1).folder, d(1).name), "VariableNamingRule", "preserve");
    names = string(T.Properties.VariableNames);
    t = getNum(T, names, ["t_after_tf_s", "t_fit_local_s"]);
    y = getNum(T, names, ["y_measured_demeaned_deg"]);
    fit = getNumOpt(T, names, ["two_mode_fit_deg"]);
    m1 = getNumOpt(T, names, ["mode1_fit_deg"]);
    m2 = getNumOpt(T, names, ["mode2_fit_deg"]);
    res = getNumOpt(T, names, ["fit_residual_deg"]);

    measuredColor = [0.12 0.12 0.12];
    fitColor = [0.88 0.18 0.14];
    mode1Color = [0.20 0.42 0.82];
    mode2Color = [0.18 0.65 0.30];
    residualColor = [0.42 0.42 0.42];

    fig = figure("Color", "w", "Position", [80 80 1550 1020]);
    tlo = tiledlayout(fig, 2, 1, "TileSpacing", "compact", "Padding", "compact");
    ax1 = nexttile(tlo); hold(ax1, "on"); grid(ax1, "on");
    hMeasured = plot(ax1, t, y, "-", "Color", measuredColor, ...
        "LineWidth", 1.1, "DisplayName", "Measured");
    hFit = plot(ax1, t, fit, "--", "Color", fitColor, ...
        "LineWidth", 1.5, "DisplayName", "Two-mode fit");

    ax2 = nexttile(tlo); hold(ax2, "on"); grid(ax2, "on");
    plot(ax2, t, m1, "-", "Color", mode1Color, ...
        "LineWidth", 1.3, "DisplayName", "Mode 1");
    plot(ax2, t, m2, "-", "Color", mode2Color, ...
        "LineWidth", 1.3, "DisplayName", "Mode 2");
    plot(ax2, t, res, "--", "Color", residualColor, ...
        "LineWidth", 1.0, "DisplayName", "Fit residual");

    % One legend serves both tiles. Proxy handles keep every legend object
    % on the same axes for compatibility across MATLAB releases.
    hMode1 = plot(ax1, nan, nan, "-", "Color", mode1Color, "LineWidth", 1.3);
    hMode2 = plot(ax1, nan, nan, "-", "Color", mode2Color, "LineWidth", 1.3);
    hResidual = plot(ax1, nan, nan, "--", "Color", residualColor, "LineWidth", 1.0);
    legend(ax1, [hMeasured hFit hMode1 hMode2 hResidual], ...
        ["Measured", "Two-mode fit", "Mode 1", "Mode 2", "Fit residual"], ...
        "Location", "northoutside", "Orientation", "horizontal", ...
        "NumColumns", 5, "Box", "off");

    ax1.XTickLabel = [];
    xlabel(tlo, "$t~(\mathrm{s})$", "Interpreter", "latex");
    ylabel(tlo, "Swing angle (deg)");
    linkaxes([ax1 ax2], "x");
    tFinite = t(isfinite(t));
    if ~isempty(tFinite)
        xlim(ax1, [min(tFinite) max(tFinite)]);
    end

    drawnow;
    [zoomT, zoomY, zoomFit, zoomLabel] = loadQuantizationDetail( ...
        BESTFIT_DIR, t, y, fit);
    addQuantizationInset(fig, ax1, zoomT, zoomY, zoomFit, 0.09, ...
        measuredColor, fitColor, zoomLabel);

    exportSvg(fig, outSvg);
    close(fig);
    fprintf("Saved %s\n", outSvg);
end

function [t, measured, fit, sourceLabel] = loadQuantizationDetail( ...
    BESTFIT_DIR, fallbackT, fallbackMeasured, fallbackFit)
    t = fallbackT;
    measured = fallbackMeasured;
    fit = fallbackFit;
    sourceLabel = "Pulse, L=1.30 m";

    patterns = {
        "bestfit_*_L090_Nonrobust_rep01_*.csv"
        "bestfit_*_L090_*Nonrobust*rep01*.csv"
        "bestfit_*_L090_*nonrobust*rep01*.csv"
    };
    d = [];
    for i = 1:numel(patterns)
        d = dir(fullfile(BESTFIT_DIR, patterns{i}));
        if ~isempty(d)
            break;
        end
    end
    if isempty(d)
        warning("L=0.90 m nonrobust rep01 decomposition not found; quantization lens uses the pulse run.");
        return;
    end

    Q = readtable(fullfile(d(1).folder, d(1).name), ...
        "VariableNamingRule", "preserve");
    names = string(Q.Properties.VariableNames);
    qT = getNum(Q, names, ["t_after_tf_s", "t_fit_local_s"]);
    qMeasured = getNum(Q, names, ["y_measured_demeaned_deg"]);
    qFit = getNumOpt(Q, names, ["two_mode_fit_deg"]);
    if any(isfinite(qT) & isfinite(qMeasured) & isfinite(qFit))
        t = qT;
        measured = qMeasured;
        fit = qFit;
        sourceLabel = "Nonrobust, L=0.90 m";
        fprintf("Quantization lens source: %s\n", fullfile(d(1).folder, d(1).name));
    else
        warning("The nonrobust quantization source lacks finite fit data; using the pulse run.");
    end
end

function addQuantizationInset(fig, parentAx, t, measured, fit, ...
    quantDeg, measuredColor, fitColor, sourceLabel)
    valid = isfinite(t) & isfinite(measured) & isfinite(fit);
    if nnz(valid) < 10
        return;
    end

    tv = t(valid);
    yv = measured(valid);
    fv = fit(valid);
    % Quantization is most visible near a low-amplitude extremum, where the
    % physical motion changes slowly and adjacent encoder counts form
    % horizontal plateaus. Prefer the latest such extremum that still
    % contains at least two observed encoder levels.
    df = diff(fv);
    extrema = find(df(1:end-1).*df(2:end) <= 0) + 1;
    usable = extrema(tv(extrema) >= tv(1) + 0.30*(tv(end)-tv(1)) & ...
        tv(extrema) <= tv(1) + 0.90*(tv(end)-tv(1)));
    halfWidth = min(0.34, 0.10*(tv(end)-tv(1)));
    centerIdx = [];
    for k = numel(usable):-1:1
        candidate = usable(k);
        candidateMask = abs(tv-tv(candidate)) <= halfWidth;
        levels = unique(round(yv(candidateMask)/quantDeg));
        changes = nnz(diff(round(yv(candidateMask)/quantDeg)) ~= 0);
        if numel(levels) >= 2 && numel(levels) <= 10 && changes >= 2
            centerIdx = candidate;
            break;
        end
    end
    if isempty(centerIdx)
        lateIdx = find(tv >= tv(1) + 0.55*(tv(end)-tv(1)));
        if isempty(lateIdx)
            lateIdx = (1:numel(tv))';
        end
        [~, localPeak] = max(abs(fv(lateIdx)));
        centerIdx = lateIdx(localPeak);
    end

    tLo = max(tv(1), tv(centerIdx)-halfWidth);
    tHi = min(tv(end), tv(centerIdx)+halfWidth);
    zoomMask = tv >= tLo & tv <= tHi;
    if nnz(zoomMask) < 5
        return;
    end

    tz = tv(zoomMask);
    yz = yv(zoomMask);
    fz = fv(zoomMask);

    % Create a compact circular lens in the upper-right of the main axes.
    % The traces are mapped to normalized lens coordinates and clipped to
    % the circle, so the inset has no rectangular plot box.
    oldUnits = parentAx.Units;
    parentAx.Units = "normalized";
    p = parentAx.Position;
    parentAx.Units = oldUnits;
    oldFigUnits = fig.Units;
    fig.Units = "pixels";
    figPixels = fig.Position;
    fig.Units = oldFigUnits;
    lensWidth = 0.13;
    lensHeight = lensWidth * figPixels(3) / max(figPixels(4), 1);
    insetPos = [p(1)+p(3)-lensWidth-0.012, ...
        p(2)+p(4)-lensHeight-0.010, lensWidth, lensHeight];
    axInset = axes(fig, "Position", insetPos, "Tag", "quantizationInset"); %#ok<LAXES>
    hold(axInset, "on");
    axis(axInset, "equal");
    xlim(axInset, [-1.04 1.04]); ylim(axInset, [-1.04 1.04]);
    set(axInset, "Visible", "off", "Color", "none");

    tMid = 0.5*(tLo+tHi);
    tHalf = max(0.5*(tHi-tLo), eps);
    yMin = min([yz; fz]);
    yMax = max([yz; fz]);
    yPad = max(0.55*quantDeg, 0.10*(yMax-yMin+eps));
    yMid = 0.5*(yMin+yMax);
    yHalf = max(0.5*(yMax-yMin+2*yPad), quantDeg);

    % Show the encoder count levels explicitly inside the lens.
    observedLevels = unique(round(yz/quantDeg)*quantDeg);
    for i = 1:numel(observedLevels)
        yn = 0.86*(observedLevels(i)-yMid)/yHalf;
        if abs(yn) < 0.92
            xExtent = sqrt(0.92^2-yn^2);
            plot(axInset, [-xExtent xExtent], [yn yn], "-", ...
                "Color", [0.86 0.86 0.86], "LineWidth", 0.55);
        end
    end

    [stairsT, stairsY] = makeStairPath(tz, yz);
    stairsXn = 0.90*(stairsT-tMid)/tHalf;
    stairsYn = 0.86*(stairsY-yMid)/yHalf;
    stairsInside = stairsXn.^2 + stairsYn.^2 <= 0.92^2;
    stairsXn(~stairsInside) = NaN;
    stairsYn(~stairsInside) = NaN;
    plot(axInset, stairsXn, stairsYn, "-", ...
        "Color", measuredColor, "LineWidth", 1.15);

    fitXn = 0.90*(tz-tMid)/tHalf;
    fitYn = 0.86*(fz-yMid)/yHalf;
    fitInside = fitXn.^2 + fitYn.^2 <= 0.92^2;
    fitXn(~fitInside) = NaN;
    fitYn(~fitInside) = NaN;
    plot(axInset, fitXn, fitYn, "-", ...
        "Color", fitColor, "LineWidth", 1.2);

    th = linspace(0, 2*pi, 300);
    plot(axInset, 0.98*cos(th), 0.98*sin(th), "k-", "LineWidth", 1.0);
    text(axInset, 0, 0.73, sprintf("%.2f^\\circ/count", quantDeg), ...
        "HorizontalAlignment", "center", "FontName", "Times New Roman", ...
        "FontSize", 8, "FontWeight", "bold");
    text(axInset, 0, -0.72, sourceLabel, ...
        "HorizontalAlignment", "center", "FontName", "Times New Roman", ...
        "FontSize", 7.2);
end

function [stairsT, stairsY] = makeStairPath(t, y)
    t = t(:);
    y = y(:);
    n = numel(t);
    if n < 2
        stairsT = t;
        stairsY = y;
        return;
    end
    stairsT = zeros(2*n-1, 1);
    stairsY = zeros(2*n-1, 1);
    stairsT(1:2:end) = t;
    stairsY(1:2:end) = y;
    stairsT(2:2:end) = t(2:end);
    stairsY(2:2:end) = y(1:end-1);
end

function addSharedTauLine(fig, axesList, tauS)
    axesList = axesList(isgraphics(axesList, 'axes'));
    if isempty(axesList) || ~isfinite(tauS)
        return;
    end
    xLimits = xlim(axesList(1));
    if tauS < xLimits(1) || tauS > xLimits(2)
        return;
    end

    positions = nan(numel(axesList), 4);
    for i = 1:numel(axesList)
        oldUnits = axesList(i).Units;
        axesList(i).Units = "normalized";
        positions(i,:) = axesList(i).Position;
        axesList(i).Units = oldUnits;
    end

    xFraction = (tauS-xLimits(1)) / diff(xLimits);
    xFigure = positions(1,1) + xFraction*positions(1,3);
    yBottom = min(positions(:,2));
    yTop = max(positions(:,2)+positions(:,4));
    annotation(fig, "line", [xFigure xFigure], [yBottom yTop], ...
        "Color", [0.85 0.08 0.08], "LineStyle", "--", ...
        "LineWidth", 2.2, "Tag", "sharedTauLine");
    annotation(fig, "textbox", ...
        [min(xFigure+0.004, 0.965), yTop-0.030, 0.030, 0.030], ...
        "String", "$\tau$", "Interpreter", "latex", ...
        "FontName", "Times New Roman", "FontSize", 14, ...
        "FontWeight", "bold", "Color", [0.85 0.08 0.08], ...
        "EdgeColor", "none", "Margin", 0, ...
        "HorizontalAlignment", "left", "VerticalAlignment", "bottom", ...
        "Tag", "sharedTauLabel");
end

function addTauLine(ax, tauS)
    yl = ylim(ax);
    if isfinite(tauS)
        xline(ax, tauS, "--", "$\tau$", "Interpreter", "latex", ...
            "Color", [0.85 0.08 0.08], "LineWidth", 2.0, ...
            "FontSize", 13, "FontWeight", "bold", ...
            "LabelVerticalAlignment", "bottom", "HandleVisibility", "off");
    end
    ylim(ax, yl);
end

function switches = detectSwitchTimes(t, cmd)
    if isempty(t) || isempty(cmd) || ~any(isfinite(cmd))
        switches = [];
        return;
    end
    cmd = fillmissing(cmd, "nearest");
    dcmd = [0; abs(diff(cmd(:)))];
    idx = find(dcmd > 1e-6);
    sw = t(idx);
    switches = [];
    for i = 1:numel(sw)
        if isempty(switches) || abs(sw(i) - switches(end)) > 0.04
            switches(end+1,1) = sw(i); %#ok<AGROW>
        end
    end
end

function tf = detectStopTime(t, cmd)
    tf = NaN;
    if isempty(t) || isempty(cmd) || ~any(isfinite(cmd)); return; end
    maxCmd = max(abs(cmd), [], "omitnan");
    thresh = max(1e-6, 0.05 * maxCmd);
    seen = false;
    for i = 1:numel(cmd)
        if abs(cmd(i)) > thresh; seen = true; end
        if seen && abs(cmd(i)) <= thresh
            tf = t(i);
            return;
        end
    end
end

function s = readSummary(summaryFile)
    s = containers.Map("KeyType", "char", "ValueType", "double");
    if ~exist(summaryFile, "file"); return; end
    txt = fileread(summaryFile);
    lines = regexp(txt, "\r?\n", "split");
    for i = 1:numel(lines)
        line = string(strtrim(lines{i}));
        tok = regexp(line, "^([A-Za-z0-9_]+):\s*([-+0-9.eE]+)", "tokens", "once");
        if ~isempty(tok)
            s(char(tok{1})) = str2double(tok{2});
        end
    end
end

function v = getSummaryValue(s, key)
    if isKey(s, key)
        v = s(key);
    else
        v = NaN;
    end
end

function methods = normalizeMethodStrings(methods)
    out = strings(size(methods));
    for i = 1:numel(methods)
        m = lower(string(methods(i)));
        if contains(m, "pulse")
            out(i) = "Pulse";
        elseif contains(m, "nonrobust")
            out(i) = "Nonrobust";
        elseif contains(m, "robust")
            out(i) = "Robust";
        elseif contains(m, "zero") || contains(m, "isa") || contains(m, "is2")
            out(i) = "AIS";
        else
            out(i) = string(methods(i));
        end
    end
    methods = out;
end

function tag = lowerMethodTag(method)
    method = string(method);
    if method == "AIS"
        tag = "ais";
    elseif method == "Nonrobust"
        tag = "nonrobust";
    elseif method == "Robust"
        tag = "robust";
    else
        tag = "pulse";
    end
end

function x = getNum(T, names, candidates)
    x = getNumOpt(T, names, candidates);
    if ~any(isfinite(x))
        error("Missing required column: %s", strjoin(candidates, ", "));
    end
end

function x = getNumOpt(T, names, candidates)
    x = nan(height(T), 1);
    candidates = string(candidates);
    for i = 1:numel(candidates)
        c = candidates(i);
        if any(names == c)
            x = forceNumeric(T.(c));
            return;
        end
    end
end

function x = forceNumeric(x)
    if isnumeric(x) || islogical(x)
        x = double(x); x = x(:); return;
    end
    if iscell(x); x = string(x); end
    if isstring(x) || ischar(x) || iscategorical(x)
        x = str2double(string(x)); x = x(:); return;
    end
    x = nan(numel(x), 1);
end

function x = resizeToN(x, n)
    x = x(:);
    if numel(x) == n; return; end
    if isempty(x)
        x = nan(n,1);
    elseif numel(x) > n
        x = x(1:n);
    else
        x(end+1:n,1) = NaN;
    end
end

function v = firstFiniteScalar(x)
    x = x(isfinite(x));
    if isempty(x)
        v = NaN;
    else
        v = x(1);
    end
end

function printRunInventory(runs, LENGTH_ORDER, METHOD_ORDER)
    fprintf("\n=== RUN INVENTORY ===\n");
    fprintf("length_m");
    for im = 1:numel(METHOD_ORDER); fprintf("\t%s", METHOD_ORDER(im)); end
    fprintf("\n");
    for iL = 1:numel(LENGTH_ORDER)
        L = LENGTH_ORDER(iL);
        fprintf("%.2f", L);
        for im = 1:numel(METHOD_ORDER)
            n = nnz(abs(runs.length_m-L)<1e-9 & runs.method==METHOD_ORDER(im));
            fprintf("\t%d", n);
        end
        fprintf("\n");
    end
end

function printAndSaveAngleStatistics(rows, LENGTH_ORDER, METHOD_ORDER, OUT_DIR)
    metrics = ["rms_demeaned_deg", "p2p_deg", "rms_deg"];
    labels = ["RMS demeaned", "Peak-to-peak", "RMS with offset"];
    stats = table();
    fprintf("\n=== RESIDUAL ANGLE STATISTICS: 0.20 s GATE ===\n");
    for iL = 1:numel(LENGTH_ORDER)
        L = LENGTH_ORDER(iL);
        for im = 1:numel(METHOD_ORDER)
            method = METHOD_ORDER(im);
            idx = abs(rows.length_m-L)<1e-9 & rows.method==method;
            row = table(); row.length_m = L; row.method = method; row.n = nnz(idx);
            fprintf("L=%.2f %-10s n=%d", L, method, row.n);
            for k = 1:numel(metrics)
                x = rows.(metrics(k))(idx);
                [mu, sd, med, lo, hi] = finiteStats(x);
                row.("mean_"+metrics(k)) = mu;
                row.("std_"+metrics(k)) = sd;
                row.("median_"+metrics(k)) = med;
                row.("min_"+metrics(k)) = lo;
                row.("max_"+metrics(k)) = hi;
                fprintf(" | %s %.5g +/- %.4g deg (median %.5g)", labels(k), mu, sd, med);
            end
            fprintf("\n");
            stats = [stats; row]; %#ok<AGROW>
        end
    end
    outCsv = fullfile(OUT_DIR, "paper_residual_angle_statistics.csv");
    writetable(stats, outCsv);
    fprintf("Saved angle statistics: %s\n", outCsv);
end

function printAndSaveEnergyStatistics(modal, LENGTH_ORDER, METHOD_ORDER, OUT_DIR)
    metric = "mode1_Eavg_reported_J_per_kg";
    requireColumn(modal, metric);
    stats = table();
    fprintf("\n=== MASS-NORMALIZED MODE-1 RESIDUAL ENERGY: 0.20 s GATE ===\n");
    for iL = 1:numel(LENGTH_ORDER)
        L = LENGTH_ORDER(iL);
        pulse = modal.(metric)(abs(modal.length_m-L)<1e-9 & modal.method=="Pulse");
        pulseMean = mean(pulse(isfinite(pulse)));
        for im = 1:numel(METHOD_ORDER)
            method = METHOD_ORDER(im);
            idx = abs(modal.length_m-L)<1e-9 & modal.method==method;
            x = modal.(metric)(idx);
            [mu, sd, med, lo, hi] = finiteStats(x);
            reduction = 100*(1-mu/pulseMean);
            row = table(L, method, nnz(isfinite(x)), mu, sd, med, lo, hi, reduction, ...
                'VariableNames', ["length_m","method","n","mean_J_per_kg","std_J_per_kg", ...
                "median_J_per_kg","min_J_per_kg","max_J_per_kg","reduction_vs_pulse_pct"]);
            stats = [stats; row]; %#ok<AGROW>
            fprintf("L=%.2f %-10s n=%d | mean %.6g +/- %.4g J/kg | median %.6g | reduction %.3f%%\n", ...
                L, method, row.n, mu, sd, med, reduction);
        end
    end
    outCsv = fullfile(OUT_DIR, "paper_residual_energy_statistics.csv");
    writetable(stats, outCsv);
    fprintf("Saved energy statistics: %s\n", outCsv);
end

function printAndSaveIdStatistics(idRows, OUT_DIR)
    if isempty(idRows)
        warning("No selected AIS identification rows were available for statistics.");
        return;
    end
    lengths = unique(idRows.length_m, "sorted");
    stats = table();
    fprintf("\n=== ONLINE SYSTEM-ID FULL-PERIOD STATISTICS ===\n");
    for i = 1:numel(lengths)
        L = lengths(i);
        idx = abs(idRows.length_m-L)<1e-9;
        [Tmu, Tsd, Tmed, Tmin, Tmax] = finiteStats(idRows.selected_id_Tosc_s(idx));
        [tmu, tsd, tmed] = finiteStats(idRows.selected_id_time_s(idx));
        [lmu, lsd, lmed] = finiteStats(idRows.lock_time_s(idx));
        [emu, esd, emed] = finiteStats(idRows.error_pct(idx));
        referenceTosc = idRows.reference_Tosc_s(find(idx,1));
        row = table(L, nnz(idx), referenceTosc, Tmu, Tsd, Tmed, Tmin, Tmax, emu, esd, emed, ...
            tmu, tsd, tmed, lmu, lsd, lmed, ...
            'VariableNames', ["length_m","n","reference_Tosc_s","mean_selected_Tosc_s","std_selected_Tosc_s", ...
            "median_selected_Tosc_s","min_selected_Tosc_s","max_selected_Tosc_s","mean_error_pct", ...
            "std_error_pct","median_error_pct","mean_selected_id_time_s","std_selected_id_time_s", ...
            "median_selected_id_time_s","mean_lock_time_s","std_lock_time_s","median_lock_time_s"]);
        stats = [stats; row]; %#ok<AGROW>
        fprintf("L=%.2f n=%d | reference Tosc=%.5f s | selected Tosc=%.5f +/- %.5f s | error=%+.3f +/- %.3f%% | selected at %.3f s | locked at %.3f s\n", ...
            L, row.n, referenceTosc, Tmu, Tsd, emu, esd, tmu, lmu);
    end
    outRows = fullfile(OUT_DIR, "paper_online_id_selected_runs.csv");
    outStats = fullfile(OUT_DIR, "paper_online_id_statistics.csv");
    writetable(idRows, outRows); writetable(stats, outStats);
    fprintf("Saved ID rows: %s\nSaved ID statistics: %s\n", outRows, outStats);
end

function printModalExampleStatistics(modal)
    examples = {
        1.30, "Pulse", 1
    };
    fprintf("\n=== SELECTED MODAL-DECOMPOSITION EXAMPLES ===\n");
    for i = 1:size(examples,1)
        L = examples{i,1}; method = string(examples{i,2}); rep = examples{i,3};
        idx = abs(modal.length_m-L)<1e-9 & modal.method==method & modal.rep==rep;
        if ~any(idx)
            fprintf("L=%.2f %s rep%02d: no modal row found\n", L, method, rep);
            continue;
        end
        r = modal(find(idx,1),:);
        fprintf("L=%.2f %-10s rep%02d | A1=%.5g deg | f1=%.5g Hz | zeta1=%.5g | reported E1/m=%.6g J/kg | fit RMSE=%.5g deg | mode2 quantization artifact=%d\n", ...
            L, method, rep, r.A1_abs_deg, r.f1_damped_Hz, r.zeta1, ...
            r.mode1_Eavg_reported_J_per_kg, r.two_mode_rmse_deg, r.mode2_quant_artifact);
    end
end

function [mu, sd, med, lo, hi] = finiteStats(x)
    x = x(isfinite(x));
    if isempty(x)
        mu = NaN; sd = NaN; med = NaN; lo = NaN; hi = NaN;
        return;
    end
    mu = mean(x); sd = std(x); med = median(x); lo = min(x); hi = max(x);
end

function p = getLastPathPart(folder)
    [~, p] = fileparts(folder);
end

function requireColumn(T, name)
    if ~ismember(name, string(T.Properties.VariableNames))
        error("Required column '%s' is missing from the modal results CSV.", name);
    end
end

function exportSvg(fig, outSvg)
    set(fig, "Renderer", "painters");

    [outDir, baseName, ~] = fileparts(outSvg);
    isHalfWidth = startsWith(baseName, "timeseries_") || ...
        startsWith(baseName, "modal_decomposition_") || ...
        startsWith(baseName, "residual_angle_");
    isOnlineId = strcmp(baseName, "online_zero_zeta_id_convergence");
    isPrimaryEnergy = strcmp(baseName, "residual_energy_primary");
    if isOnlineId
        tickFontSize = 10;
        xLabelFontSize = 11.5;
        yLabelFontSize = 11;
        panelTitleFontSize = 13;
        legendFontSize = 10;
    elseif isPrimaryEnergy
        tickFontSize = 11;
        xLabelFontSize = 12;
        yLabelFontSize = 11;
        panelTitleFontSize = 12.5;
        legendFontSize = 10;
    elseif isHalfWidth
        tickFontSize = 13;
        xLabelFontSize = 15;
        yLabelFontSize = 15;
        panelTitleFontSize = 15;
        legendFontSize = 11;
    else
        tickFontSize = 11;
        xLabelFontSize = 13;
        yLabelFontSize = 13;
        panelTitleFontSize = 14;
        legendFontSize = 10;
    end

    ax = findall(fig, "Type", "axes");
    for i = 1:numel(ax)
        if strcmp(ax(i).Tag, "quantizationInset")
            set(ax(i), "FontName", "Times New Roman", ...
                "FontSize", max(8, tickFontSize-4), ...
                "LineWidth", 0.7, "Box", "on");
            continue;
        end
        set(ax(i), "FontName", "Times New Roman", "FontSize", tickFontSize, ...
            "LineWidth", 0.8, "Box", "on");
        set(ax(i).XLabel, "FontName", "Times New Roman", "FontSize", xLabelFontSize);
        set(ax(i).YLabel, "FontName", "Times New Roman", "FontSize", yLabelFontSize);
        set(ax(i).Title, "FontName", "Times New Roman", "FontSize", panelTitleFontSize);
    end
    legends = findall(fig, "Type", "legend");
    for i = 1:numel(legends)
        set(legends(i), "FontName", "Times New Roman", ...
            "FontSize", legendFontSize);
        if ~isempty(legends(i).Title.String)
            set(legends(i).Title, "FontName", "Times New Roman", ...
                "FontSize", panelTitleFontSize, "FontWeight", "bold");
        end
    end

    % Normalize the vector canvas so LaTeX scaling is predictable. The
    % moderate font preset above sits between the previous oversized 16/14 pt
    % setting and MATLAB's too-small 10/9 pt defaults.
    pixelPos = fig.Position;
    aspect = pixelPos(4) / max(pixelPos(3), 1);
    fig.Units = "inches";
    inchPos = fig.Position;
    inchPos(3) = 7.2;
    inchPos(4) = 7.2 * aspect;
    fig.Position = inchPos;
    set(fig, "PaperPositionMode", "auto");
    drawnow;

    outPdf = fullfile(outDir, baseName + ".pdf");
    outPng = fullfile(outDir, baseName + ".png");

    % PDF is the publication master. MATLAB writes the text and line work
    % directly into the PDF, avoiding the font substitution and text-position
    % problems that can occur when Overleaf converts MATLAB SVG through
    % Inkscape.
    try
        exportgraphics(fig, outPdf, "ContentType", "vector", ...
            "BackgroundColor", "white");
    catch
        print(fig, outPdf, "-dpdf", "-painters", "-bestfit");
    end

    % Keep a high-resolution raster fallback for journals or systems that
    % reject complex vector graphics.
    try
        exportgraphics(fig, outPng, "Resolution", 600, ...
            "BackgroundColor", "white");
    catch
        print(fig, outPng, "-dpng", "-r600");
    end

    % Retain SVG only as an editable secondary artifact. Do not prefer it in
    % LaTeX when the PDF exists.
    try
        exportgraphics(fig, outSvg, "ContentType", "vector", "BackgroundColor", "white");
    catch
        % Compatibility path for MATLAB releases without SVG support in
        % exportgraphics. The painters renderer keeps lines and text vector.
        print(fig, outSvg, "-dsvg", "-painters");
    end

    fprintf("  PDF master: %s\n", outPdf);
    fprintf("  PNG fallback: %s\n", outPng);
end
