using Microsoft.Extensions.DependencyInjection;
using SpawnDev.UnitTesting;
using SpawnDev.VoxelEngine.DemoConsole;

var services = new ServiceCollection();
var sp = services.BuildServiceProvider();
var runner = new UnitTestRunner(sp, true);
await ConsoleRunner.Run(args, runner);
