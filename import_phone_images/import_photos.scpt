tell application "Image Capture"
    set devList to every device
    repeat with dev in devList
        if name of dev contains "iPhone" then
            download every image of dev to POSIX file "/Users/yourname/Downloads/iPhoneImages"
        end if
    end repeat
end tell
