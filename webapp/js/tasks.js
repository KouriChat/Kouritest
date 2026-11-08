/* Task metadata mirrors experiment_config/tasks.json and quant/quantify.py MAPS. */
(function () {
  'use strict';

  const GLOBAL_CELLS = 16200;

  const TASKS = {
    global: {
      key: 'global', label: '全球 2°', short: '全球',
      rows: 90, cols: 180, upscale: [1440, 720],
      latTop: 89, latBottom: -89, lonLeft: -179, lonRight: 179,
      archiveSuffix: ''
    },
    csc: {
      key: 'csc', label: '南海 0.2°', short: '南海',
      rows: 111, cols: 141, upscale: [1128, 888],
      latTop: 23, latBottom: 1, lonLeft: 99, lonRight: 127,
      archiveSuffix: '_csc'
    },
    dzh: {
      key: 'dzh', label: '地中海 0.2°', short: '地中海',
      rows: 71, cols: 231, upscale: [1848, 568],
      latTop: 45, latBottom: 31, lonLeft: -9, lonRight: 37,
      archiveSuffix: '_dzh'
    },
    seto: {
      key: 'seto', label: '濑户内海 2/63°', short: '濑户内海',
      rows: 64, cols: 253, upscale: [2024, 512],
      latTop: 35, latBottom: 33, lonLeft: 129, lonRight: 137,
      archiveSuffix: '_seto'
    },
    arctic01: {
      key: 'arctic01', label: '北极航道 0.1°', short: '北极航道',
      rows: 61, cols: 241, upscale: [1928, 488],
      latTop: 79, latBottom: 73, lonLeft: -105, lonRight: -81,
      archiveSuffix: '_arc01'
    },
    hecate: {
      key: 'hecate', label: '赫卡特海峡 2/126°', short: '赫卡特',
      rows: 127, cols: 127, upscale: [1016, 1016],
      latTop: 55, latBottom: 53, lonLeft: -131, lonRight: -129,
      archiveSuffix: '_hec'
    }
  };

  const TASK_ORDER = ['global', 'csc', 'dzh', 'seto', 'arctic01', 'hecate'];

  function cells(task) { return task.rows * task.cols; }

  function normalized(task, correct) {
    return correct * GLOBAL_CELLS / cells(task);
  }

  function stepText(task) {
    if (task.key === 'global') return '2°';
    if (task.key === 'csc' || task.key === 'dzh') return '0.2°';
    if (task.key === 'seto') return '2/63°';
    if (task.key === 'arctic01') return '0.1°';
    return '2/126°';
  }

  function boundsText(task) {
    const ns = (v) => (v >= 0 ? v + 'N' : (-v) + 'S');
    const ew = (v) => (v >= 0 ? v + 'E' : (-v) + 'W');
    return ns(task.latTop) + '–' + ns(task.latBottom) + '，' +
      ew(task.lonLeft) + '–' + ew(task.lonRight);
  }

  window.MapTasks = {
    GLOBAL_CELLS, TASKS, TASK_ORDER, cells, normalized, stepText, boundsText
  };
})();