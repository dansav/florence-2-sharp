using System;
using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using FlorenceTwoLab.Desktop.ViewModels;

namespace FlorenceTwoLab.Desktop.Views;

public partial class MainView : UserControl
{
    public MainView()
    {
        InitializeComponent();
        
        DataContext = new MainViewModel();

        Loaded += MainWindow_Loaded;
        
        AddHandler(DragDrop.DragOverEvent, DragOver);
        AddHandler(DragDrop.DragLeaveEvent, DragLeave);
        AddHandler(DragDrop.DropEvent, Drop);
    }
    
    private async void MainWindow_Loaded(object? sender, RoutedEventArgs e)
    {
        if (DataContext is MainViewModel vm)
        {
            try
            {
                await vm.InitializeAsync();
            }
            catch (Exception exception)
            {
                Debug.WriteLine(exception);
                throw;
            }
        }
    }
    
    private async void Drop(object? sender, DragEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;
        
        vm.IsPreviewVisible = true;

        if (!e.Data.Contains(DataFormats.Files)) return;
        
        var items = e.Data.GetFiles() ?? Array.Empty<IStorageItem>();
        foreach (var item in items)
        {
            if (item is IStorageFile file)
            {
                await using var stream = await file.OpenReadAsync();
                await vm.LoadImageAsync(stream);
                return;
            }
                
            if (item is IStorageFolder folder)
            {
                await foreach (var item2 in folder.GetItemsAsync())
                {
                    if (item2 is IStorageFile file2)
                    {
                        await using var stream = await file2.OpenReadAsync();
                        await vm.LoadImageAsync(stream);
                        return;
                    }
                }
            }
        }
    }

    private void DragOver(object? sender, DragEventArgs e)
    {
        if (DataContext is MainViewModel vm)
        {
            vm.IsPreviewVisible = false;
        }
    }
    
    private void DragLeave(object? sender, DragEventArgs e)
    {
        if (DataContext is MainViewModel vm)
        {
            vm.IsPreviewVisible = true;
        }
    }
}