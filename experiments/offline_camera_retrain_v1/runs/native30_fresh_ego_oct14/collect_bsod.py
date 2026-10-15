"""Read Windows Event Log over WSL interop; no system configuration changes."""
import datetime
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def main():
    command = r'''
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
    $start=(Get-Date).AddDays(-2)
    $system=@(Get-WinEvent -FilterHashtable @{LogName='System';Id=41,1001,1074;StartTime=$start} |
      ForEach-Object { [PSCustomObject]@{Time=$_.TimeCreated.ToString('o');Id=$_.Id;Provider=$_.ProviderName;Message=$_.Message} })
    $application=@(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1001;StartTime=$start} -MaxEvents 100 -ErrorAction SilentlyContinue |
      Where-Object {$_.Message -match 'BlueScreen|LiveKernelEvent'} |
      ForEach-Object { [PSCustomObject]@{Time=$_.TimeCreated.ToString('o');Message=$_.Message} })
    $drivers=@(Get-CimInstance Win32_PnPSignedDriver | Where-Object {$_.DeviceClass -eq 'DISPLAY'} |
      Select-Object DeviceName,DriverVersion)
    $dumpReadable=$false
    try {$s=[System.IO.File]::OpenRead('C:\Windows\Minidump\101526-13109-01.dmp'); $s.Close();$dumpReadable=$true} catch {}
    [PSCustomObject]@{LastBoot=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o');System=$system;Application=$application;DisplayDrivers=$drivers;DumpReadable=$dumpReadable} | ConvertTo-Json -Depth 5
    '''
    result = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', command],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    raw = json.loads(result.stdout.decode('utf-8-sig'))
    events = []
    for event in raw['System']:
        if event['Id'] != 1001:
            continue
        code = re.search(r'0x[0-9a-fA-F]+', event['Message'])
        dump = re.search(r'\d{6}-\d+-\d+\.dmp', event['Message'])
        basename = dump.group() if dump else None
        bucket = None
        for app in raw['Application']:
            if basename and basename in app['Message']:
                match = re.search(r'AV_[^\s，,]+', app['Message'])
                if match:
                    bucket = match.group()
                    break
        events.append(dict(recovery_event_time=event['Time'], bugcheck=code.group() if code else None,
            minidump=basename, WER_failure_bucket=bucket))
    report = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
        status='Windows host bugchecks confirmed; originating driver/hardware unresolved',
        current_Windows_boot=raw['LastBoot'], events=events,
        event41_times=[r['Time'] for r in raw['System'] if r['Id'] == 41],
        event1074_times=[r['Time'] for r in raw['System'] if r['Id'] == 1074],
        display_drivers=raw['DisplayDrivers'], minidump_readable=raw['DumpReadable'],
        limitations='Non-elevated access denies minidump and WER Report.wer. Failure bucket is crash location, not proof of originating faulty driver. Event timestamps are recovery/report times, not exact crash times. No automatic driver update, reboot, permissions or training changes.',
        next_diagnostic='Administrator-readable copies of both Oct15 minidumps and WinDbg !analyze -v / stack / module review; correlate network filters for tcpip bucket without blaming tcpip alone.',
        references=['https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0xd1--driver-irql-not-less-or-equal',
            'https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x50--page-fault-in-nonpaged-area'],
        system_changes_performed=False)
    (ROOT / 'BSOD_DIAGNOSIS_OCT15.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
