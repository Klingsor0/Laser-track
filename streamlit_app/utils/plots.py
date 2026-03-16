import matplotlib.pyplot as plt
from matplotlib import rcParams

#matplotlib.use("pgf")
#matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'sans serif',
#    'font.size' : 7,
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#})
# quiero una funcion para exportar y guardar la figura 
def plot_save_fig_profile(x_obs, y_obs, x_dom, function, filename, file_format):
    fluorescent_green_rgb = (57/255,255/255,20/255)
    plt.figure(figsize=(7.5/2,3.5/2))
    plt.gca().set_aspect('equal')
    plt.grid(color = "white")
    plt.title(r"Perfil de superficie libre reflejada por la guía onda",y=1.04, wrap=True) 
    plt.xlabel(r'$r/R_s$') 
    plt.ylabel(r'$y/2\gamma R$') 
    plt.scatter(x_obs, y_obs, marker = 'o', color = "black", label = "Puntos procesados de fotografía", s=0.2) 
    #etiqueta2A = f"$ y = ({'{:0.3E}'.format(pendiente2A)}) x + {'{:0.3E}'.format(intercepcion2A)}$"
    plt.plot(x_dom, function(x_dom), color = fluorescent_green_rgb, label = f"Curva deducida ") 
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend(prop={'size': 7})
    plt.savefig(filename, format=file_format, dpi=1200)

def plot_save_fig_deflection(x_obs, y_obs,  y_error, x_dom, function, filename_out):
    fluorescent_green_rgb = (57/255,255/255,20/255)
    plt.figure(figsize=(3,2))
    plt.grid(color = "white")
    plt.title(r"Ángulo de deflexión en función de la distancia al centro",y=1.04, wrap=True)
    plt.xlabel(r'$b/R_s$')
    plt.ylabel(r'$\alpha\, \, (rad)$')
    plt.errorbar(x_obs, y_obs, yerr=y_err, fmt='^', color = "black", label = "Datos experimentales a 150 RPM", capsize=2)
    #plt.scatter(x_angulo, angulo_obs, marker = '^', color = "black", label = "Datos experimentales a 150 RPM")
    #etiqueta2A = f"$ y = ({'{:0.3E}'.format(pendiente2A)}) x + {'{:0.3E}'.format(intercepcion2A)}$"
    plt.plot(x_seq, angulo(x_seq), color = fluorescent_green_rgb, label = f"Curva estimada por geometría de superficie libre")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend(prop={'size': 6})
    plt.savefig('../../Tesis/Figuras/grafica_datos_angulo.pgf', format='pgf', dpi=1200)

def plot_save_fig_profile_PR(x_obs, y_obs, x_dom, function, filename, file_format):
    """
    Create publication-quality plot following Physical Review style guidelines
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    # Physical Review style parameters
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 9
    rcParams['mathtext.fontset'] = 'stix'  # Matches Times
    rcParams['axes.linewidth'] = 0.8
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.minor.width'] = 0.6
    rcParams['ytick.minor.width'] = 0.6
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    
    # Physical Review column width: 3.4 inches (single) or 7.0 inches (double)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))  # Single column, 4:3 aspect
    
    # Plot data - black for observations (PR standard)
    ax.scatter(x_obs, y_obs, 
              marker='o', 
              s=2.0,  # Slightly larger for visibility
              color='black', 
              alpha=0.6,
              label='Processed data',
              rasterized=True)  # Rasterize for smaller file size
    
    # Plot model - solid line (avoid fluorescent colors in print)
    ax.plot(x_dom, function(x_dom), 
           color='#CC0000',  # Dark red (PR-friendly, distinct from black)
           linewidth=1.2, 
           label='Theoretical curve',
           zorder=10)
    
    # Labels with proper LaTeX
    ax.set_xlabel(r'$r/R_{\mathrm{s}}$', fontsize=10)
    ax.set_ylabel(r'$y/(2\gamma R)$', fontsize=10)
    
    # Title (optional - PR often prefers captions over titles)
    # ax.set_title('Free surface profile', fontsize=10, pad=8)
    
    # Grid (subtle, or omit for PR)
    ax.grid(True, which='major', color='#CCCCCC', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)  # Grid behind data
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=8, 
                   top=True, right=True)  # Ticks on all sides
    ax.tick_params(axis='both', which='minor', 
                   top=True, right=True)
    ax.minorticks_on()
    
    # Legend
    ax.legend(frameon=True, 
             fancybox=False, 
             edgecolor='black',
             fontsize=8,
             loc='best',
             framealpha=1.0)
    
    # Scientific notation
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2), useMathText=True)
    
    # Tight layout
    plt.tight_layout(pad=0.3)
    
    # Save with PR specifications
    # PR prefers: EPS for vector, TIFF for raster, 300+ DPI
    if file_format.lower() in ['eps', 'pdf']:
        # Vector formats
        plt.savefig(filename, 
                   format=file_format, 
                   dpi=600,  # High DPI even for vector (rasterized elements)
                   bbox_inches='tight',
                   pad_inches=0.02)
    else:
        # Raster formats (TIFF, PNG)
        plt.savefig(filename, 
                   format=file_format, 
                   dpi=600,  # PR minimum is 300, 600 is better
                   bbox_inches='tight',
                   pad_inches=0.02)
    
    plt.close()
