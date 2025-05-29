# Heat Transfer MCP Server Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/heat-transfer-mcp.git
cd heat-transfer-mcp
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import ht; print(f'HT version: {ht.__version__}')"
python -c "from thermo import Chemical; print('Thermo library OK')"
python -c "import pint; print('Unit conversion OK')"
```

### 5. Run the Server

```bash
python server.py
```

You should see:
```
✓ UNIT CONVERSION: ENABLED
  Supported units:
    Temperature: °F, °C, K
    Length: ft, in, m
    Flow Rate: GPM, SCFM, lb/hr, kg/s
    ...
```

## Claude Desktop Integration

### 1. Find Claude Desktop Config Location

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Mac:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### 2. Add Heat Transfer Server

Edit the config file to include:

```json
{
  "mcpServers": {
    "heat-transfer": {
      "command": "python",
      "args": ["C:\\path\\to\\heat-transfer-mcp\\server.py"]
    }
  }
}
```

### 3. Restart Claude Desktop

The Heat Transfer tools will now be available in Claude.

## Testing the Installation

Run the example scripts:

```bash
python examples/basic_usage.py
python examples/wastewater_applications.py
```

## Troubleshooting

### Issue: ImportError for 'ht' module
**Solution:** Ensure virtual environment is activated and reinstall:
```bash
pip install --upgrade ht
```

### Issue: Unit conversion not working
**Solution:** Check pint installation:
```bash
pip install --upgrade pint
```

### Issue: Material properties showing fallback
**Solution:** This is normal - the server will use HT's insulation module when available, fallback otherwise.

## Common Usage Patterns

### Imperial Units (Automatic Conversion)
```python
temperature="95 degF"    # Converts to 308.15 K
flow_rate="200 GPM"      # Converts to 12.62 kg/s
pressure="14.7 psi"      # Converts to 101325 Pa
length="10 feet"         # Converts to 3.048 m
```

### SI Units (Direct Use)
```python
temperature=308.15       # K
flow_rate=12.62         # kg/s
pressure=101325         # Pa
length=3.048            # m
```

## Support

For issues or questions:
1. Check the documentation in the `docs/` folder
2. Review examples in the `examples/` folder
3. Submit an issue on GitHub