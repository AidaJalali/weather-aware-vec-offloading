
(.venv) alireza@alireza-mlk:~/Sharif/Term 8/Real Time Systems/Project/Phase2/weather-aware-vec-offloading$ git merge origin/feature/sumo-mobility
Updating 8be0769..059eb55
Fast-forward
 .gitignore                                                       |    4 +
 README.md                                                        |  130 +-
 scripts/generate_all_datasets.py                                 |  122 ++
 source/compare_reward_profiles.py                                |   12 +-
 source/data/hard_tasks/chunk_0.xml                               |  123 --
 source/data/raw_mobility.xml                                     | 2043 --------------------
 source/data/sumo/train/s111_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3806297 bytes
 source/data/sumo/train/s111_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 523200 bytes
 source/data/sumo/train/s123_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3801915 bytes
 source/data/sumo/train/s123_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 523380 bytes
 source/data/sumo/train/s123_u5_f1_vl/tasks/chunk_0.xml.gz        |  Bin 0 -> 1570290 bytes
 source/data/sumo/train/s123_u5_f1_vl/vehicles/chunk_0.xml.gz     |  Bin 0 -> 209695 bytes
 source/data/sumo/train/s123_u8_f2_l/tasks/chunk_0.xml.gz         |  Bin 0 -> 2509723 bytes
 source/data/sumo/train/s123_u8_f2_l/vehicles/chunk_0.xml.gz      |  Bin 0 -> 346446 bytes
 source/data/sumo/train/s222_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3806518 bytes
 source/data/sumo/train/s222_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 524092 bytes
 source/data/sumo/train/s333_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3794140 bytes
 source/data/sumo/train/s333_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 525225 bytes
 source/data/sumo/train/s42_u12_f3/tasks/chunk_0.xml.gz           |  Bin 0 -> 3795820 bytes
 source/data/sumo/train/s42_u12_f3/vehicles/chunk_0.xml.gz        |  Bin 0 -> 523996 bytes
 source/data/sumo/train/s42_u20_f5_h/tasks/chunk_0.xml.gz         |  Bin 0 -> 6377940 bytes
 source/data/sumo/train/s42_u20_f5_h/vehicles/chunk_0.xml.gz      |  Bin 0 -> 873780 bytes
 source/data/sumo/train/s42_u30_f7_vh/tasks/chunk_0.xml.gz        |  Bin 0 -> 9538573 bytes
 source/data/sumo/train/s42_u30_f7_vh/vehicles/chunk_0.xml.gz     |  Bin 0 -> 1371712 bytes
 source/data/sumo/train/s444_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3818563 bytes
 source/data/sumo/train/s444_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 523963 bytes
 source/data/sumo/train/s456_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3808559 bytes
 source/data/sumo/train/s456_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 523217 bytes
 source/data/sumo/train/s555_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3798409 bytes
 source/data/sumo/train/s555_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 524968 bytes
 source/data/sumo/train/s666_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3800902 bytes
 source/data/sumo/train/s666_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 523357 bytes
 source/data/sumo/train/s789_u12_f3/tasks/chunk_0.xml.gz          |  Bin 0 -> 3791288 bytes
 source/data/sumo/train/s789_u12_f3/vehicles/chunk_0.xml.gz       |  Bin 0 -> 524074 bytes
 source/data/sumo/train/weather_base_s100/tasks/chunk_0.xml.gz    |  Bin 0 -> 1148997 bytes
 source/data/sumo/train/weather_base_s100/vehicles/chunk_0.xml.gz |  Bin 0 -> 586355 bytes
 source/data/sumo/train/weather_base_s200/tasks/chunk_0.xml.gz    |  Bin 0 -> 1007472 bytes
 source/data/sumo/train/weather_base_s200/vehicles/chunk_0.xml.gz |  Bin 0 -> 586376 bytes
 source/data/sumo/train/weather_base_s300/tasks/chunk_0.xml.gz    |  Bin 0 -> 999197 bytes
 source/data/sumo/train/weather_base_s300/vehicles/chunk_0.xml.gz |  Bin 0 -> 585988 bytes
 source/data/sumo/train/weather_fog_s100/tasks/chunk_0.xml.gz     |  Bin 0 -> 3844631 bytes
 source/data/sumo/train/weather_fog_s100/vehicles/chunk_0.xml.gz  |  Bin 0 -> 521334 bytes
 source/data/sumo/train/weather_fog_s200/tasks/chunk_0.xml.gz     |  Bin 0 -> 3856729 bytes
 source/data/sumo/train/weather_fog_s200/vehicles/chunk_0.xml.gz  |  Bin 0 -> 523313 bytes
 source/data/sumo/train/weather_fog_s300/tasks/chunk_0.xml.gz     |  Bin 0 -> 3847824 bytes
 source/data/sumo/train/weather_fog_s300/vehicles/chunk_0.xml.gz  |  Bin 0 -> 521848 bytes
 source/data/sumo/train/weather_rain_s100/tasks/chunk_0.xml.gz    |  Bin 0 -> 1695369 bytes
 source/data/sumo/train/weather_rain_s100/vehicles/chunk_0.xml.gz |  Bin 0 -> 574538 bytes
 source/data/sumo/train/weather_rain_s200/tasks/chunk_0.xml.gz    |  Bin 0 -> 1509141 bytes
 source/data/sumo/train/weather_rain_s200/vehicles/chunk_0.xml.gz |  Bin 0 -> 573649 bytes
 source/data/sumo/train/weather_rain_s300/tasks/chunk_0.xml.gz    |  Bin 0 -> 1504486 bytes
 source/data/sumo/train/weather_rain_s300/vehicles/chunk_0.xml.gz |  Bin 0 -> 572631 bytes
 source/data/sumo/train/weather_snow_s100/tasks/chunk_0.xml.gz    |  Bin 0 -> 2453461 bytes
 source/data/sumo/train/weather_snow_s100/vehicles/chunk_0.xml.gz |  Bin 0 -> 557579 bytes
 source/data/sumo/train/weather_snow_s200/tasks/chunk_0.xml.gz    |  Bin 0 -> 2458415 bytes
 source/data/sumo/train/weather_snow_s200/vehicles/chunk_0.xml.gz |  Bin 0 -> 557600 bytes
 source/data/sumo/train/weather_snow_s300/tasks/chunk_0.xml.gz    |  Bin 0 -> 2460123 bytes
 source/data/sumo/train/weather_snow_s300/vehicles/chunk_0.xml.gz |  Bin 0 -> 556866 bytes
 source/data/sumo/val/s999_u12_f3/tasks/chunk_0.xml.gz            |  Bin 0 -> 3786037 bytes
 source/data/sumo/val/s999_u12_f3/vehicles/chunk_0.xml.gz         |  Bin 0 -> 523371 bytes
 source/data/sumo/val/weather_base_s999/tasks/chunk_0.xml.gz      |  Bin 0 -> 1001150 bytes
 source/data/sumo/val/weather_base_s999/vehicles/chunk_0.xml.gz   |  Bin 0 -> 585666 bytes
 source/data/sumo/val/weather_fog_s999/tasks/chunk_0.xml.gz       |  Bin 0 -> 3845164 bytes
 source/data/sumo/val/weather_fog_s999/vehicles/chunk_0.xml.gz    |  Bin 0 -> 521424 bytes
 source/data/sumo/val/weather_rain_s999/tasks/chunk_0.xml.gz      |  Bin 0 -> 1491709 bytes
 source/data/sumo/val/weather_rain_s999/vehicles/chunk_0.xml.gz   |  Bin 0 -> 572636 bytes
 source/data/sumo/val/weather_snow_s999/tasks/chunk_0.xml.gz      |  Bin 0 -> 2455020 bytes
 source/data/sumo/val/weather_snow_s999/vehicles/chunk_0.xml.gz   |  Bin 0 -> 555540 bytes
 source/data/tasks/chunk_0.xml                                    | 5008 --------------------------------------------------
 source/data/vehicles/chunk_0.xml                                 | 2043 --------------------
 source/infrastructure.py                                         |   24 +-
 source/random_baseline_simulator.py                              |   34 +-
 source/run_phase1_pipeline.py                                    |  119 +-
 source/sumo_pipeline.py                                          |  338 ++++
 source/sumo_traci_smoke_test.py                                  |   79 +
 source/task_generation.py                                        |  137 ++
 source/weather_scenario_generator.py                             |    6 +-
 source/xml_dataset_writer.py                                     |  211 +++
 tests/test_task_generation.py                                    |  143 ++
 tests/test_xml_dataset_writer.py                                 |  204 ++
 80 files changed, 1436 insertions(+), 9344 deletions(-)
 create mode 100644 scripts/generate_all_datasets.py
 delete mode 100644 source/data/hard_tasks/chunk_0.xml
 delete mode 100644 source/data/raw_mobility.xml
 create mode 100644 source/data/sumo/train/s111_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s111_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s123_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s123_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s123_u5_f1_vl/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s123_u5_f1_vl/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s123_u8_f2_l/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s123_u8_f2_l/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s222_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s222_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s333_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s333_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s42_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s42_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s42_u20_f5_h/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s42_u20_f5_h/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s42_u30_f7_vh/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s42_u30_f7_vh/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s444_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s444_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s456_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s456_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s555_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s555_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s666_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s666_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s789_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/s789_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_base_s100/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_base_s100/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_base_s200/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_base_s200/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_base_s300/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_base_s300/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_fog_s100/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_fog_s100/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_fog_s200/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_fog_s200/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_fog_s300/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_fog_s300/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_rain_s100/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_rain_s100/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_rain_s200/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_rain_s200/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_rain_s300/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_rain_s300/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_snow_s100/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_snow_s100/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_snow_s200/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_snow_s200/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_snow_s300/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/train/weather_snow_s300/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/s999_u12_f3/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/s999_u12_f3/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_base_s999/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_base_s999/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_fog_s999/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_fog_s999/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_rain_s999/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_rain_s999/vehicles/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_snow_s999/tasks/chunk_0.xml.gz
 create mode 100644 source/data/sumo/val/weather_snow_s999/vehicles/chunk_0.xml.gz
 delete mode 100644 source/data/tasks/chunk_0.xml
 delete mode 100644 source/data/vehicles/chunk_0.xml
 create mode 100644 source/sumo_pipeline.py
 create mode 100644 source/sumo_traci_smoke_test.py
 create mode 100644 source/task_generation.py
 create mode 100644 source/xml_dataset_writer.py
 create mode 100644 tests/test_task_generation.py
 create mode 100644 tests/test_xml_dataset_writer.py