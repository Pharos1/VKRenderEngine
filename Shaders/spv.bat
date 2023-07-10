::C:/VulkanSDK/1.3.250.0/Bin/glslc.exe main.vert -o spv/main.vert.spv
::C:/VulkanSDK/1.3.250.0/Bin/glslc.exe main.frag -o spv/main.frag.spv

@echo off
for /r %%i in (*.frag, *.vert) do %VULKAN_SDK%/Bin/glslc %%i -o %%~pi/spv/%%~ni%%~xi.spv 
:: source https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-xp/bb490909(v=technet.10)?redirectedfrom=MSDN