platform :ios, '14.0'
use_frameworks!

target 'IronCount' do
  pod 'TensorFlowLiteSwift'
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.shell_script_build_phases.each do |phase|
      if phase.name.include?("Copy Pods Resources")
        phase.shell_script = phase.shell_script.gsub(
          "realpath -m",
          "python3 -c \"import os,sys; print(os.path.abspath(sys.argv[1]))\""
        )
      end
    end
  end
end