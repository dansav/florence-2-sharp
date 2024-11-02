using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Avalonia.Controls;

namespace FlorenceTwoLab.Desktop;

public class ConsoleHelper : TextWriter
{
    private readonly TextWriter _original;
    private readonly TextBlock _screen;

    public ConsoleHelper(TextWriter original, TextBlock screen)
    {
        _original = original;
        _screen = screen;
    }

    public override Encoding Encoding => _original.Encoding;

    public override void Write(char value)
    {
        Debug.Write(value);
        _original.Write(value);
        _screen.Text += value;
    }

    public override void Write(string? value)
    {
        Debug.Write(value);
        _original.Write(value);
        _screen.Text += value;
    }

    public override void WriteLine()
    {
        Debug.WriteLine("");
        _original.WriteLine();
        _screen.Text += Environment.NewLine;
    }

    public override void WriteLine(string? value)
    {
        Debug.WriteLine(value);
        _original.WriteLine(value);
        _screen.Text += value + Environment.NewLine;
    }
}
