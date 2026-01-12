final: prev: {
  pythonPackagesExtensions = prev.pythonPackagesExtensions or [ ] ++ [
    (finalPythonPackages: prevPythonPackages: {
      nix-cuda-sr = finalPythonPackages.callPackage ./nix-cuda-sr { };
      plotly = prevPythonPackages.plotly.overridePythonAttrs (prevAttrs: {
        # TODO: These also fail on aarch64-darwin -- maybe it just fails on aarch64?
        # python3.12-plotly> FAILED plotly/tests/test_optional/test_kaleido/test_kaleido.py::test_kaleido_engine_to_image_returns_bytes - ValueError: Failed to start Kaleido subprocess. Error stream:
        # python3.12-plotly> FAILED plotly/tests/test_optional/test_kaleido/test_kaleido.py::test_kaleido_fulljson - ValueError: Failed to start Kaleido subprocess. Error stream:
        # python3.12-plotly> FAILED plotly/tests/test_optional/test_kaleido/test_kaleido.py::test_bytesio - ValueError: Failed to start Kaleido subprocess. Error stream:
        disabledTestPaths =
          prevAttrs.disabledTestPaths
          ++ final.lib.optionals final.stdenv.hostPlatform.isAarch64 [
            "plotly/tests/test_optional/test_kaleido"
          ];
      });
      optuna = prevPythonPackages.optuna.overridePythonAttrs (prevAttrs: {
        # So the transitive dependency on OpenCV isn't brought in, which fails to build with
        # CUDA 12.9.
        doCheck = false;
        
        # TODO: These also fail on aarch64-darwin -- maybe it just fails on aarch64?
        disabledTests =
          prevAttrs.disabledTests
          ++ final.lib.optionals final.stdenv.hostPlatform.isAarch64 [
            # ValueError: Failed to start Kaleido subprocess. Error stream
            # kaleido/executable/kaleido: line 5:  5956 Illegal instruction: 4  ./bin/kaleido $@
            "test_get_optimization_history_plot"
            "test_plot_intermediate_values"
            "test_plot_rank"
            "test_plot_terminator_improvement"

            # Fatal Python error: Aborted
            # matplotlib/backend_bases.py", line 2654 in create_with_canvas
            "test_edf_plot_no_trials"
            "test_get_timeline_plot"
            "test_plot_contour"
            "test_plot_contour_customized_target_name"
            "test_plot_edf_with_multiple_studies"
            "test_plot_edf_with_target"
            "test_plot_parallel_coordinate"
            "test_plot_parallel_coordinate_customized_target_name"
            "test_plot_param_importances"
            "test_plot_param_importances_customized_target_name"
            "test_plot_param_importances_multiobjective_all_objectives_displayed"
            "test_plot_slice"
            "test_plot_slice_customized_target_name"
            "test_target_is_none_and_study_is_multi_obj"
            "test_visualizations_with_single_objectives"
          ];
      });
    })
  ];
}
