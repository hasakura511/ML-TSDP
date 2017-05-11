//  Performance Chart

zingchart.MODULESDIR = "https://cdn.zingchart.com/modules/";
ZC.LICENSE = ["569d52cefae586f634c54f86dc99e6a9", "ee6b7db5b51705a13dc2339db3edaf6d"];

var tab_value_index = []

$(document).on('click', '.chart-button', function(event) {

    var get_all_tab='';
    var ij = 0;

    $( ".chart-pane-tab" ).each(function() {
        get_all_tab= $(".chart-tab-icon-text" ).text();
    });

    arr = get_all_tab.split('K');

    for(i=0; i < arr.length; i++)
        tab_value_index[i] = arr[i];

});


$(document).on('click', '.chart-pane-tab', function(event) {

    var board_value = $('.chart-title-text').html();
    board_value=board_value.split(":");
    board_value=board_value[1].replace(/\s+/g, '');

    var tab_value=$(this).text();
    tab_value = tab_value.replace(/\s+/g, '');

    var anti_val = '';

    $.ajax
    ({
        url: '/getchartdata',
        type: 'get',
        dataType: 'json', // added data type
        success: function(result)
        {
          var date = [];
          var system = [];
          var anti_system = [];
          var benchmark = [];
          var cumper = [];

          var anti_system_cum = [];
          var benchmark_cum = [];

          var chart_title = '';

          if(tab_value==tab_value_index[2] + "K")
          {

            chart_title = "v4micro 20Days Historical Performance: "+board_value+", Anti-"+board_value+", Benchmark";
            $.each(result.v4micro_performance, function(l,m) {
              if(l==board_value)
              {
                  $.each(this, function(k, v) {

                      date.push(k);
                      system.push(v);

                  });
              }

              if(board_value=='AntiHighestEquity' && l=='HighestEquity')
              {
                  anti_val = 'HighestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='AntiHighestEquity' && board_value=='HighestEquity')
              {
                  anti_val = 'AntiHighestEquity';
                 $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='Previous' && l=='Anti-Previous')
              {
                  anti_val = 'Anti-Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='Previous' && board_value=='Anti-Previous')
              {
                  anti_val = 'Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='RiskOn' && l=='RiskOff')
              {
                  anti_val = 'RiskOff';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='RiskOn' && board_value=='RiskOff')
              {
                  anti_val = 'RiskOn';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='50/50' && l=='Anti50/50')
              {
                  anti_val = 'Anti50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='50/50' && board_value=='Anti50/50')
              { 
                  anti_val = '50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='LowestEquity' && l=='AntiLowestEquity')
              {
                  anti_val = 'AntiLowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='LowestEquity' && board_value=='AntiLowestEquity')
              {
                  anti_val = 'LowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }


              else if(board_value=='Seasonality' && l=='Anti-Seasonality')
              {
                  anti_val = 'Anti-Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='Seasonality' && board_value=='Anti-Seasonality')
              {
                  anti_val = 'Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else
              {
                 if(l=='Anti-'+board_value)
                  {
                      anti_val = 'Anti-'+ board_value;
                      $.each(this, function(k, v) {

                          anti_system.push(v);

                      });
                  }
                  if(l=='Anti-'+board_value+'_cumper')
                  {
                      anti_system_cum_val = 'Anti-'+ board_value+'_Cum %';
                      $.each(this, function(k, v) {

                      	  v = v + " %";
                          anti_system_cum.push(v);

                      });
                  }
              }
             
              if(l=='benchmark')
              {
                  $.each(this, function(k, v) {

                      benchmark.push(v);

                  });
              }
              if(l=='benchmark_cumper')
              {
                  $.each(this, function(k, v) {

                  	  v = v + " %";
                      benchmark_cum.push(v);

                  });
              }
              if(l==board_value+"_cumper")
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      cumper.push(v);

                  });
              }
            });
          }
          else if(tab_value==tab_value_index[1] + "K")
          {
            chart_title = "v4mini 20Days Historical Performance: "+board_value+", Anti-"+board_value+", Benchmark";
            $.each(result.v4mini_performance, function(l,m) {
              if(l==board_value)
              {
                  $.each(this, function(k, v) {

                      date.push(k);
                      system.push(v);

                  });
              }


              if(board_value=='AntiHighestEquity' && l=='HighestEquity')
              {
                  anti_val = 'HighestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='AntiHighestEquity' && board_value=='HighestEquity')
              {
                  anti_val = 'AntiHighestEquity';
                 $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='Previous' && l=='Anti-Previous')
              {
                  anti_val = 'Anti-Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='Previous' && board_value=='Anti-Previous')
              {
                  anti_val = 'Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='RiskOn' && l=='RiskOff')
              {
                  anti_val = 'RiskOff';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='RiskOn' && board_value=='RiskOff')
              {
                  anti_val = 'RiskOn';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='50/50' && l=='Anti50/50')
              {
                  anti_val = 'Anti50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='50/50' && board_value=='Anti50/50')
              { 
                  anti_val = '50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='LowestEquity' && l=='AntiLowestEquity')
              {
                  anti_val = 'AntiLowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='LowestEquity' && board_value=='AntiLowestEquity')
              {
                  anti_val = 'LowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }


              else if(board_value=='Seasonality' && l=='Anti-Seasonality')
              {
                  anti_val = 'Anti-Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='Seasonality' && board_value=='Anti-Seasonality')
              {
                  anti_val = 'Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else
              {
                 if(l=='Anti-'+board_value)
                  {
                      anti_val = 'Anti-'+ board_value;
                      $.each(this, function(k, v) {

                          anti_system.push(v);

                      });
                  }
                  if(l=='Anti-'+board_value+'_cumper')
                  {
                      anti_system_cum_val = 'Anti-'+ board_value+'_Cum %';
                      $.each(this, function(k, v) {

                          anti_system_cum.push(v);

                      });
                  }
              }


              if(l=='benchmark')
              {
                  $.each(this, function(k, v) {

                      benchmark.push(v);

                  });
              }
              if(l=='benchmark_cumper')
              {
                  $.each(this, function(k, v) {

                      benchmark_cum.push(v);

                  });
              }

              if(l==board_value+"_cumper")
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      cumper.push(v);

                  });
              }
            });
          }
           else if(tab_value==tab_value_index[0] + "K")
          {
            chart_title = "v4futures 20Days Historical Performance: "+board_value+", Anti-"+board_value+", Benchmark";
            $.each(result.v4futures_performance, function(l,m) {
              if(l==board_value)
              {
                  $.each(this, function(k, v) {

                      date.push(k);
                      system.push(v);

                  });
              }
              if(board_value=='AntiHighestEquity' && l=='HighestEquity')
              {
                  anti_val = 'HighestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='AntiHighestEquity' && board_value=='HighestEquity')
              {
                  anti_val = 'AntiHighestEquity';
                 $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='Previous' && l=='Anti-Previous')
              {
                  anti_val = 'Anti-Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='Previous' && board_value=='Anti-Previous')
              {
                  anti_val = 'Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='RiskOn' && l=='RiskOff')
              {
                  anti_val = 'RiskOff';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='RiskOn' && board_value=='RiskOff')
              {
                  anti_val = 'RiskOn';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='50/50' && l=='Anti50/50')
              {
                  anti_val = 'Anti50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='50/50' && board_value=='Anti50/50')
              { 
                  anti_val = '50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }



              else if(board_value=='LowestEquity' && l=='AntiLowestEquity')
              {
                  anti_val = 'AntiLowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='LowestEquity' && board_value=='AntiLowestEquity')
              {
                  anti_val = 'LowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }


              else if(board_value=='Seasonality' && l=='Anti-Seasonality')
              {
                  anti_val = 'Anti-Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else if(l=='Seasonality' && board_value=='Anti-Seasonality')
              {
                  anti_val = 'Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
              }
              else
              {
                 if(l=='Anti-'+board_value)
                  {
                      anti_val = 'Anti-'+ board_value;
                      $.each(this, function(k, v) {

                          anti_system.push(v);

                      });
                  }
                  if(l=='Anti-'+board_value+'_cumper')
                  {
                      anti_system_cum_val = 'Anti-'+ board_value+'_Cum %';
                      $.each(this, function(k, v) {

                          anti_system_cum.push(v);

                      });
                  }
              }

              
              if(l=='benchmark')
              {
                  $.each(this, function(k, v) {

                      benchmark.push(v);

                  });
              }
              if(l=='benchmark_cumper')
              {
                  $.each(this, function(k, v) {

                      benchmark_cum.push(v);

                  });
              }
              if(l==board_value+"_cumper")
              {
                  $.each(this, function(k, v) {
                      v = v + " %";
                      cumper.push(v);

                  });
              }
            });
          }
          else
          {
            console.log("NO BOARD");
            // ruturn false;
          }

          var max_system = Math.max.apply(Math,system);
          var min_system = Math.min.apply(Math,system);

          var max_anti_system = Math.max.apply(Math,anti_system);
          var min_anti_system = Math.min.apply(Math,anti_system);

          var max_benchmark = Math.max.apply(Math,benchmark);
          var min_benchmark = Math.min.apply(Math,benchmark);

          var max = Math.max(max_system, max_anti_system, max_benchmark);
          var min = Math.min(min_system, min_anti_system, min_benchmark);
          
          max = max + 5000;
          min = min - 5000;

          var tot='';
          tot+=min.toString();
          tot+=":";
          tot+=max.toString();
          tot+=":1000";

          
      
          var myConfig =  {
            "type":"line",
            "utc": true,
            "title": {
              "text": chart_title,
              "font-size": "14px",
              "adjust-layout": true
            },
            "plotarea": {
              "margin": "dynamic 45 60 dynamic",
            },
            "scale-x":{
              "values":date,
              "transform": {
                "type": "date",
                "all": "%d %M %Y",
                "guide": {
                  "visible": false
                },
              },
              "item":{  
                "font-angle":315,  
              } 
            },
            "scale-y":{
              "values":tot,
              "guide": {
                "line-style": "dashed"
              },
              "thousands-separator":",",
            },
            "series":[
              {"values":system,
              "line-color":"#0000ff",
              "line-style":"line",
              "text": board_value,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":anti_system,
              "line-color":"#0B850C",
              "line-style":"line",
              "text": anti_val,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":benchmark,
              "line-color":"#ff0000",
              "line-style":"line",
              "text": "Benchmark",
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":cumper,
              "line-color":"#ff0000",
              "line-style":"line",
              "text": "Cum %",
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":anti_system_cum,
              "line-color":"#ff0000",
              "line-style":"line",
              "text": anti_system_cum_val,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":benchmark_cum,
              "line-color":"#ff0000",
              "line-style":"line",
              "text": "Benchmark Cum %",
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
            ],
            "plot": {
              "highlight": true,
              "tooltip-text": "%t: %v<br>Date:%k",
              "shadow": 0,
              "line-width": "2px",
              "marker": {
                "type": "circle",
                "size": 1
              },
              "highlight-state": {
                "line-width": 3
              },
              "animation": {
                "effect": 1,
                "sequence": 2,
                "speed": 100,
              },
            },
            "crosshair-x": {
              "line-color": "#efefef",
              "plot-label": {
                "border-radius": "5px",
                "border-width": "1px",
                "border-color": "#f6f7f8",
                "padding": "10px",
                "font-weight": "bold",
                "thousands-separator":",",
              },
              "scale-label": {
                "font-color": "#000",
                "background-color": "#f6f7f8",
                "border-radius": "5px"
              },
            },
            "tooltip": {
              "visible": false
            },
          };

          zingchart.render({
            id: 'performance_chart',
            data: myConfig,
          });
        }
    });
});